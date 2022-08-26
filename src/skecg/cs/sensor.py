import cr.sparse as crs
import cr.sparse.lop


def build_sensor(key, m, n, type='rademacher'):
    Phi = crs.lop.rademacher_dict(key, m, n, 
        normalize_atoms=False)
    Phi = crs.lop.jit(Phi)
    return Phi
