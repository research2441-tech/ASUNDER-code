"""Minimal directive-driven HLS: unrolling, resource-constrained list scheduling,
functional-unit and register binding, FSM emission to Verilog.

Directive vector for one replica:
  u      loop unroll factor (divisor of the trip count)
  alloc  functional units per operation class (1 or 2)
  share  register binding: 'le' (left-edge sharing) or 'ded' (one register per value)
  fsm    state encoding: 'bin' or 'onehot'
  sched  list-scheduling direction: 'asap' or 'alap'
"""
import itertools
from kernels import AES_SBOX, PRESENT_SBOX, PERM16, output_layout, input_layout

FREE = {
    'ror:7': lambda e, w: (f"{{{e}[6:0],{e}[15:7]}}", 16),
    'rol:2': lambda e, w: (f"{{{e}[13:0],{e}[15:14]}}", 16),
    'shl8w16': lambda e, w: (f"{{{e}[7:0],8'd0}}", 16),
    'zext16': lambda e, w: (f"{{{16 - w}'d0,{e}}}", 16),
    'zext10': lambda e, w: (f"{{{10 - w}'d0,{e}}}", 10),
    'shl1w10': lambda e, w: (f"{{{e}[8:0],1'b0}}", 10),
    'perm16': lambda e, w: ("{" + ",".join(f"{e}[{PERM16.index(15 - k)}]" for k in range(16)) + "}", 16),
    'msb12': lambda e, w: (f"{e}[11]", 1),
    'notmsb12': lambda e, w: (f"~{e}[11]", 1),
}

FU_EXPR = {
    'add': "fa + fb", 'sub': "fa - fb", 'xor': "fa ^ fb", 'and': "fa & fb", 'or': "fa | fb",
    'mul': "fa * fb", 'sbox8': "f_sbox8(fa[7:0])", 'sbox4x4': "f_sbox4x4(fa[15:0])",
    'xtime': "{fa[6:0],1'b0} ^ (fa[7] ? 8'h1b : 8'h00)",
    'crcstep': "{fa[14:0],1'b0} ^ (fa[15] ? 16'h1021 : 16'h0000)",
    'sra': "$signed(fa) >>> fb[3:0]", 'addsub': "fc[0] ? (fa - fb) : (fa + fb)",
}
NARGS = {'add': 2, 'sub': 2, 'xor': 2, 'and': 2, 'or': 2, 'mul': 2, 'sbox8': 1, 'sbox4x4': 1,
         'xtime': 1, 'crcstep': 1, 'sra': 2, 'addsub': 3}

DIRECTIVE_SPACE = dict(u=[1, 2, 4], alloc=[1, 2], share=['le', 'ded'], fsm=['bin', 'onehot'],
                       sched=['asap', 'alap'])


def all_configs(k):
    us = [u for u in DIRECTIVE_SPACE['u'] if k['N'] % u == 0]
    out = []
    for u, a, s, f, d in itertools.product(us, DIRECTIVE_SPACE['alloc'], DIRECTIVE_SPACE['share'],
                                           DIRECTIVE_SPACE['fsm'], DIRECTIVE_SPACE['sched']):
        out.append(dict(u=u, alloc=a, share=s, fsm=f, sched=d))
    return out


def cfg_key(c):
    return f"u{c['u']}_a{c['alloc']}_{c['share']}_{c['fsm']}_{c['sched']}"


def clog2(n):
    b = 0
    while (1 << b) < n:
        b += 1
    return max(b, 1)


class Replica:
    def __init__(self, k, cfg):
        self.k, self.cfg = k, cfg
        self.U = cfg['u']
        self.NS = k['N'] // self.U
        self._unroll()
        self._schedule()
        self._bind()

    # ---------------------------------------------------------------- unroll
    def _unroll(self):
        k, U = self.k, self.U
        self.ops = []          # dict(name, base, copy, op, args, w)
        nxt_prev = None
        self.copy_next = []
        for c in range(U):
            def res(ref, c=c, nxt_prev=nxt_prev):
                t = ref[0]
                if t == 'car':
                    return ('car', ref[1]) if c == 0 else nxt_prev[ref[1]]
                if t == 'op':
                    return ('op', f"{ref[1]}#{c}")
                if t in ('arr', 'carr'):
                    return (t, ref[1], ref[2], ref[3], c)
                if t == 'iter':
                    return ('iter', c)
                if t == 'free':
                    return ('free', ref[1], res(ref[2]))
                return ref
            for dst, op, args, w in k['body']:
                self.ops.append(dict(name=f"{dst}#{c}", base=dst, copy=c, op=op,
                                     args=[res(a) for a in args], w=w))
            nxt = {v: res(r) for v, r in k['next'].items()}
            self.copy_next.append(nxt)
            self.copy_outarr = getattr(self, 'copy_outarr', [])
            self.copy_outarr.append({n: (res(r), w) for n, (r, w) in k['outarr'].items()})
            nxt_prev = nxt
        self.final_next = self.copy_next[-1] if U else {}
        self.byname = {o['name']: o for o in self.ops}

    @staticmethod
    def _op_deps(ref):
        if ref[0] == 'op':
            return [ref[1]]
        if ref[0] == 'free':
            return Replica._op_deps(ref[2])
        return []

    # -------------------------------------------------------------- schedule
    def _schedule(self):
        ops, A = self.ops, self.cfg['alloc']
        preds = {o['name']: sorted({d for a in o['args'] for d in self._op_deps(a)}) for o in ops}
        succs = {o['name']: [] for o in ops}
        for n, ps in preds.items():
            for p in ps:
                succs[p].append(n)
        order = [o['name'] for o in ops]

        def longest(adj):
            memo = {}

            def f(n):
                if n not in memo:
                    memo[n] = 1 + max([f(m) for m in adj[n]], default=0)
                return memo[n]
            return {n: f(n) for n in order}
        if self.cfg['sched'] == 'asap':
            dep, prio = preds, longest(succs)
        else:
            dep, prio = succs, longest(preds)
        step, done, t = {}, set(), 0
        while len(done) < len(order):
            t += 1
            used = {}
            ready = [n for n in order if n not in done and all(d in done and step[d] < t for d in dep[n])]
            ready.sort(key=lambda n: (-prio[n], order.index(n)))
            for n in ready:
                cls = self.byname[n]['op']
                if used.get(cls, 0) < A:
                    used[cls] = used.get(cls, 0) + 1
                    step[n] = t
                    done.add(n)
        S = t
        if self.cfg['sched'] == 'alap':
            step = {n: S - s + 1 for n, s in step.items()}
        self.state_of, self.S = step, S
        self.preds = preds

    # ------------------------------------------------------------------ bind
    def _bind(self):
        S, ops = self.S, self.ops
        UPD = S + 1
        uses = {o['name']: [] for o in ops}
        for o in ops:
            for a in o['args']:
                for d in self._op_deps(a):
                    uses[d].append(self.state_of[o['name']])
        for ref in list(self.final_next.values()):
            for d in self._op_deps(ref):
                uses[d].append(UPD)
        for oa in self.copy_outarr:
            for ref, w in oa.values():
                for d in self._op_deps(ref):
                    uses[d].append(UPD)
        self.life = {}
        for o in ops:
            sp = self.state_of[o['name']]
            last = max(uses[o['name']], default=sp + 1)
            self.life[o['name']] = (sp + 1, last)
        # register binding
        names = sorted(self.life, key=lambda n: (self.life[n][0], self.life[n][1], n))
        regs = []        # list of (end, [values])
        self.reg_of = {}
        for n in names:
            s0, s1 = self.life[n]
            placed = False
            if self.cfg['share'] == 'le':
                for r in regs:
                    if r[0] < s0:
                        r[0] = s1
                        r[1].append(n)
                        self.reg_of[n] = r[2]
                        placed = True
                        break
            if not placed:
                rid = len(regs)
                regs.append([s1, [n], rid])
                self.reg_of[n] = rid
        self.regs = [(r[2], r[1], max(self.byname[v]['w'] for v in r[1])) for r in regs]
        # functional-unit binding
        self.fu_of = {}
        self.fus = {}   # cls -> count
        for s in range(1, S + 1):
            cnt = {}
            for o in ops:
                if self.state_of[o['name']] == s:
                    cls = o['op']
                    i = cnt.get(cls, 0)
                    cnt[cls] = i + 1
                    self.fu_of[o['name']] = (cls, i)
                    self.fus[cls] = max(self.fus.get(cls, 0), i + 1)

    # ------------------------------------------------------------------ emit
    def latency(self):
        # cycle index at which done first reads 1 (IDLE is cycle 0)
        return 1 + self.NS * (self.S + 1)

    def verilog(self, modname='rep'):
        k, S, U, NS = self.k, self.S, self.U, self.NS
        UPD, DONE = S + 1, S + 2
        nst = S + 3
        L = []
        inl, outl = input_layout(k), output_layout(k)
        win = sum(w for _, _, w in inl)
        wout = sum(w for _, _, w in outl)
        L.append(f"module {modname}(input clk, input [{win - 1}:0] din, output [{wout - 1}:0] dout, output done);")
        # input buses
        off = 0
        bus = {}
        for n, w in k['inputs'].items():
            L.append(f"wire [{w - 1}:0] in_{n} = din[{off + w - 1}:{off}];")
            off += w
        for n, (w, Ln) in k['arrays'].items():
            L.append(f"wire [{w * Ln - 1}:0] bus_{n} = din[{off + w * Ln - 1}:{off}];")
            bus[n] = (w, Ln)
            off += w * Ln
        # state
        if self.cfg['fsm'] == 'bin':
            sw = clog2(nst)
            L.append(f"reg [{sw - 1}:0] state = 0;")
            for s in range(nst):
                L.append(f"wire st_{s} = (state == {sw}'d{s});")
        else:
            sw = nst - 1
            L.append(f"reg [{sw - 1}:0] state = 0;")
            L.append("wire st_0 = ~|state;")
            for s in range(1, nst):
                L.append(f"wire st_{s} = state[{s - 1}];")
        cw = clog2(NS + 1)
        if NS > 1:
            L.append(f"reg [{cw - 1}:0] cnt = 0;")
        iw = clog2(k['N']) + 1
        # registers
        for rid, vals, w in self.regs:
            L.append(f"reg [{w - 1}:0] R{rid} = 0;")
        for v, (_, w) in k['carried'].items():
            L.append(f"reg [{w - 1}:0] C_{v} = 0;")
        for n, (_, w) in k['outarr'].items():
            for j in range(k['N']):
                L.append(f"reg [{w - 1}:0] O_{n}_{j} = 0;")
        # wires for references
        self._wires = {}
        self._wl = []
        self._iw = iw

        def it_expr(c):
            if NS > 1:
                return f"(({iw}'d0 + cnt) * {U} + {c})"
            return f"{iw}'d{c}"

        def wire_of(ref):
            key = repr(ref)
            if key in self._wires:
                return self._wires[key]
            t = ref[0]
            if t == 'in':
                e, w = f"in_{ref[1]}", k['inputs'][ref[1]]
            elif t == 'car':
                e, w = f"C_{ref[1]}", k['carried'][ref[1]][1]
            elif t == 'op':
                o = self.byname[ref[1]]
                e, w = f"R{self.reg_of[ref[1]]}[{o['w'] - 1}:0]", o['w']
            elif t == 'const':
                e, w = f"{ref[2]}'d{ref[1]}", ref[2]
            elif t == 'iter':
                e, w = it_expr(ref[1]), iw
            elif t == 'arr':
                _, n, st, of, c = ref
                w, Ln = bus[n]
                idx = f"((({it_expr(c)}) * {st} + {of}) & {Ln - 1})"
                e = f"bus_{n} >> ({idx} * {w})"
            elif t == 'carr':
                _, n, st, of, c = ref
                w, vals = k['carr'][n]
                idx = f"((({it_expr(c)}) * {st} + {of}) & {len(vals) - 1 if len(vals) & (len(vals) - 1) == 0 else 7})"
                e = f"f_carr_{n}({idx})"
            elif t == 'free':
                src = wire_of(ref[2])
                e, w = FREE[ref[1]](src, self._wires_w[src])
            else:
                raise ValueError(ref)
            nm = f"w{len(self._wires)}"
            self._wl.append(f"wire [{w - 1}:0] {nm} = {e};")
            self._wires[key] = nm
            self._wires_w[nm] = w
            return nm
        self._wires_w = {}
        # functional units
        fu_lines = []
        for cls, cntu in sorted(self.fus.items()):
            ops_c = [o for o in self.ops if o['op'] == cls]
            w = max(o['w'] for o in ops_c)
            for i in range(cntu):
                mine = [o for o in ops_c if self.fu_of[o['name']] == (cls, i)]
                na = NARGS[cls]
                for ai, an in enumerate(['fa', 'fb', 'fc'][:na]):
                    terms = []
                    for o in mine:
                        terms.append((self.state_of[o['name']], wire_of(o['args'][ai])))
                    expr = terms[-1][1]
                    for s, wn in reversed(terms[:-1]):
                        expr = f"st_{s} ? {wn} : {expr}"
                    fu_lines.append(f"wire [{w - 1}:0] {cls}{i}_{an} = {expr};")
                fe = FU_EXPR[cls].replace('fa', f"{cls}{i}_fa").replace('fb', f"{cls}{i}_fb").replace('fc', f"{cls}{i}_fc")
                fu_lines.append(f"wire [{w - 1}:0] {cls}{i}_y = {fe};")
        # update-state wires
        nxt_w = {v: wire_of(r) for v, r in self.final_next.items()}
        init_w = {v: wire_of(r) for v, (r, w) in k['carried'].items()}
        out_w = [{n: wire_of(r) for n, (r, w) in oa.items()} for oa in self.copy_outarr]
        L += self._wl
        L += fu_lines
        # sequential logic
        L.append("always @(posedge clk) begin")

        def goto(s):
            if self.cfg['fsm'] == 'bin':
                return f"state <= {sw}'d{s};"
            if s == 0:
                return f"state <= {sw}'d0;"
            return f"state <= {sw}'d{1 << (s - 1)};"
        L.append(f"  if (st_0) begin {goto(1)}" + (" cnt <= 0;" if NS > 1 else "") +
                 "".join(f" C_{v} <= {init_w[v]};" for v in k['carried']) + " end")
        for s in range(1, S + 1):
            L.append(f"  if (st_{s}) {goto(s + 1)}")
        upd = [f"C_{v} <= {nxt_w[v]};" for v in k['carried']]
        for c, oa in enumerate(out_w):
            for n, wn in oa.items():
                for j in range(k['N']):
                    if (j - c) % U == 0:
                        jj = (j - c) // U
                        cond = f"cnt == {jj}" if NS > 1 else "1'b1"
                        upd.append(f"if ({cond}) O_{n}_{j} <= {wn};")
        if NS > 1:
            upd.append("cnt <= cnt + 1;")
            upd.append(f"if (cnt == {NS - 1}) {goto(DONE)} else {goto(1)}")
        else:
            upd.append(goto(DONE))
        L.append(f"  if (st_{UPD}) begin " + " ".join(upd) + " end")
        L.append(f"  if (st_{DONE}) {goto(DONE)}")
        # value registers
        for rid, vals, w in self.regs:
            branches = []
            for v in vals:
                s = self.state_of[v]
                cls, i = self.fu_of[v]
                branches.append(f"if (st_{s}) R{rid} <= {cls}{i}_y;")
            L.append("  " + " else ".join(branches))
        L.append("end")
        # outputs
        parts = []
        for n, j, w in outl:
            parts.append(f"O_{n}_{j}" if j is not None else f"C_{k['outs'][n]}")
        L.append(f"assign dout = {{{', '.join(reversed(parts))}}};")
        L.append(f"assign done = st_{DONE};")
        # functions
        if 'sbox8' in self.fus:
            L.append("function [7:0] f_sbox8(input [7:0] a); case (a)")
            L += [f"  8'd{i}: f_sbox8 = 8'd{v};" for i, v in enumerate(AES_SBOX)]
            L.append("endcase endfunction")
        if 'sbox4x4' in self.fus:
            L.append("function [3:0] f_s4(input [3:0] a); case (a)")
            L += [f"  4'd{i}: f_s4 = 4'd{v};" for i, v in enumerate(PRESENT_SBOX)]
            L.append("endcase endfunction")
            L.append("function [15:0] f_sbox4x4(input [15:0] a); f_sbox4x4 = {f_s4(a[15:12]),f_s4(a[11:8]),f_s4(a[7:4]),f_s4(a[3:0])}; endfunction")
        for n, (w, vals) in k['carr'].items():
            iwc = 4
            L.append(f"function [{w - 1}:0] f_carr_{n}(input [{iwc - 1}:0] i); case (i)")
            L += [f"  {iwc}'d{i}: f_carr_{n} = {w}'d{v};" for i, v in enumerate(vals)]
            L.append(f"  default: f_carr_{n} = {w}'d0; endcase endfunction")
        L.append("endmodule")
        return "\n".join(L)

    # ---------------------------------------------------------- timelines
    def timelines(self):
        """Live intervals of DFG values held in each register bit.

        Returns dict regname -> list of (value_key, width, t_start, t_end) in absolute
        cycles (IDLE = cycle 0). Value keys identify (base operation, iteration) and are
        therefore comparable between replicas built with different directives."""
        S, U = self.S, self.U
        tl = {}

        def key_of(ref, j):
            t = ref[0]
            if t == 'op':
                o = self.byname[ref[1]]
                return ('op', o['base'], j * U + o['copy'])
            if t == 'free':
                return ('free', ref[1], key_of(ref[2], j))
            if t == 'car':
                return ('car', ref[1], j * U)
            return ('base', repr(ref))

        def canon(key, j):
            # carried value at start of iteration i == next-expression of iteration i-1
            if key[0] == 'car':
                v, it = key[1], key[2]
                if it == 0:
                    return ('init', v)
                jj, cc = divmod(it - 1, U)
                return canon(key_of(self.copy_next[cc][v], jj), jj)
            if key[0] == 'free':
                return ('free', key[1], canon(key[2], j))
            return key
        for j in range(self.NS):
            base = 1 + j * (S + 1)
            for rid, vals, w in self.regs:
                for v in vals:
                    s0, s1 = self.life[v]
                    o = self.byname[v]
                    tl.setdefault(f"R{rid}", []).append(
                        (canon(('op', o['base'], j * U + o['copy']), j), o['w'], base + s0 - 1, base + s1 - 1))
            for v, (_, w) in self.k['carried'].items():
                key = canon(('car', v, j * U), j)
                tl.setdefault(f"C_{v}", []).append((key, w, base, base + S))
        lat = self.latency()
        for n, (ref, w) in self.k['outarr'].items():
            for j in range(self.k['N']):
                jj, c = divmod(j, U)
                key = canon(key_of(self.copy_outarr[c][n][0], jj), jj)
                tw = 1 + jj * (S + 1) + S + 1
                tl.setdefault(f"O_{n}_{j}", []).append((key, w, tw, lat + 4))
        return tl

    def role(self, regname):
        if regname == 'state':
            return 'fsm'
        if regname == 'cnt':
            return 'cnt'
        if regname.startswith('R'):
            return 'val'
        if regname.startswith('C_'):
            return 'car'
        if regname.startswith('O_'):
            return 'out'
        return 'other'


WRAPPER = """module wrap(input clk, input [{W1}:0] oa, input [{W1}:0] ob, input da, input db,
            output reg [{W1}:0] q = 0, output reg alarm = 0, output reg valid = 0);
always @(posedge clk) if (da & db & ~valid) begin valid <= 1'b1; alarm <= (oa != ob); q <= oa; end
endmodule"""
