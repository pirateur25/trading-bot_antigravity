# backtest/native.py — compile kernels.c avec gcc au premier import (gcc est présent sur les runners GitHub Actions)
# et expose les fonctions via ctypes.
import ctypes as C
import os
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "kernels.c")
LIB = os.path.join(HERE, "_kernels.so")


def _build():
    if not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(SRC):
        subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", LIB, SRC, "-lm"])
    return C.CDLL(LIB)


_lib = _build()
_P = np.ctypeslib.ndpointer
f8, i4, i8, b1 = _P(np.float64, flags="C"), _P(np.int32, flags="C"), _P(np.int64, flags="C"), _P(np.int8, flags="C")

_lib.mss_events.restype = C.c_int
_lib.mss_events.argtypes = [C.c_int, f8, f8, f8, f8, C.c_int, C.c_double, C.c_int, C.c_int, C.c_double, f8, C.c_double, C.c_double,
                            C.c_int, b1, i4, i4, i4, f8, f8, f8]
_lib.mitigate.restype = None
_lib.mitigate.argtypes = [C.c_int, f8, f8, i8, C.c_int, C.c_double, C.c_int, f8, f8, i8, i8, i8]
_lib.build_signals.restype = C.c_int
_lib.build_signals.argtypes = [C.c_int, i4, i4, i4, f8, i8, f8, f8, f8, f8,
                               C.c_int, f8, f8, i8, i8, C.c_int, f8, f8, i8, i8, f8, b1, f8, f8, f8, i4]
_lib.simulate.restype = C.c_int
_lib.simulate.argtypes = [C.c_int, f8, f8, f8, f8, C.c_int, i4, f8, f8, f8,
                          C.c_double, C.c_double, C.c_double, C.c_double, C.c_double, C.c_int, b1, b1,
                          f8, i4, i4, i4, f8, f8, f8, f8, i4, f8, f8, f8]


def _c(a, dt):
    return np.ascontiguousarray(a, dtype=dt)


def mss_events(h, l, c, o, k, ratio, spmode=0, trough_pivot=0, min_sweep_pct=0.0, atr=None, min_corr_pct=0.0, min_corr_atr=0.0, max_back=5):
    n = len(h)
    ev = np.zeros(n, np.int8)
    h1i, h2i, li = (np.full(n, -1, np.int32) for _ in range(3))
    h1p, h2p, l2p = (np.full(n, np.nan) for _ in range(3))
    atr = np.zeros(n) if atr is None else atr
    _lib.mss_events(n, _c(h, np.float64), _c(l, np.float64), _c(c, np.float64), _c(o, np.float64), int(k), float(ratio),
                    int(spmode), int(trough_pivot), float(min_sweep_pct), _c(atr, np.float64), float(min_corr_pct), float(min_corr_atr),
                    int(max_back), ev, h1i, h2i, li, h1p, h2p, l2p)
    idx = np.flatnonzero(ev)
    return dict(t=idx.astype(np.int32), h1i=h1i[idx], h2i=h2i[idx], li=li[idx], h1p=h1p[idx], h2p=h2p[idx], l2p=l2p[idx])


def mitigate(zlo, zhi, form, ztype, depth, mh, ml, mopen, mclose):
    out = np.empty(len(zlo), np.int64)
    _lib.mitigate(len(zlo), _c(zlo, np.float64), _c(zhi, np.float64), _c(form, np.int64), int(ztype), float(depth),
                  len(mh), _c(mh, np.float64), _c(ml, np.float64), _c(mopen, np.int64), _c(mclose, np.int64), out)
    return out


def build_signals(ev, conf, pz, oz, P):
    ne = len(ev["t"])
    ok = np.zeros(ne, np.int8)
    entry, sl, tp = (np.zeros(ne) for _ in range(3))
    zid = np.full(ne, -1, np.int32)
    _lib.build_signals(ne, _c(ev["t"], np.int32), _c(ev["h1i"], np.int32), _c(ev["h2i"], np.int32), _c(ev["h2p"], np.float64), _c(conf["close_ns"], np.int64),
                       _c(conf["h"], np.float64), _c(conf["l"], np.float64), _c(conf["c"], np.float64), _c(conf["atr"], np.float64),
                       len(pz["lo"]), pz["lo"], pz["hi"], pz["form"], pz["mit"],
                       len(oz["lo"]), oz["lo"], oz["hi"], oz["form"], oz["mit"],
                       _c(P, np.float64), ok, entry, sl, tp, zid)
    return ok.astype(bool), entry, sl, tp, zid


def simulate(o, h, l, c, sidx, sentry, ssl, stp, capital, risk_pct, fee, lev, be_rr, ttl, noentry=None, fexit=None):
    n, ns = len(o), len(sidx)
    noentry = np.zeros(n, np.int8) if noentry is None else _c(noentry, np.int8)
    fexit = np.zeros(n, np.int8) if fexit is None else _c(fexit, np.int8)
    eq = np.empty(n)
    m = max(ns, 1)
    ti = [np.zeros(m, np.int32) for _ in range(4)]
    tf = [np.zeros(m) for _ in range(6)]
    tr = np.zeros(m, np.int32)
    nt = _lib.simulate(n, o, h, l, c, ns, _c(sidx, np.int32), _c(sentry, np.float64), _c(ssl, np.float64), _c(stp, np.float64),
                       float(capital), float(risk_pct), float(fee), float(lev), float(be_rr), int(ttl), noentry, fexit,
                       eq, ti[0], ti[1], ti[2], tf[0], tf[1], tf[2], tf[3], tr, tf[4], tf[5], np.zeros(m))
    cols = dict(sig=ti[0][:nt], fill=ti[1][:nt], exit=ti[2][:nt], entry=tf[0][:nt], xprice=tf[1][:nt], sl=tf[2][:nt], tp=tf[3][:nt],
                reason=tr[:nt], pnl=tf[4][:nt], r=tf[5][:nt])
    return cols, eq
