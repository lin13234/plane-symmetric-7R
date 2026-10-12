"""一键复现全部精确拓扑证书（SymPy 精确有理/代数数运算，无浮点判定）。

用法（在本目录）：python run_all_certificates.py
依次运行下列脚本，任一脚本非零退出即判为失败；结果汇总写入 all_certificates_summary.json。
"""
import json, subprocess, sys, time, platform
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    ("certify_coverage_identities.py", "S5-S6 global coverage identities"),
    ("certify_quartic_discriminant.py", "S6-S7 quartic, Cramer identity and discriminant from MDH parameters"),
    ("certify_s0_rotation_components.py", "S5 sin q3 = 0 rotation components for general q7"),
    ("certify_folded_planar_cycles.py", "S4.3 folded circles: closure and rank"),
    ("certify_conventional_planar_rank.py", "S9 planar-cycle rank census"),
    ("certify_spatial_regular_rank.py", "S9 spatial-cycle rank census"),
    ("certify_exceptional_spatial_points.py", "S7 exceptional atlas points"),
    ("certify_spatial_sheet_gluing.py", "S8 sheet gluing leading coefficients and Sturm counts"),
    ("certify_gateway_rank.py", "S9.1 interface closure and rank five"),
    ("certify_interface_transversality.py", "S11 Lemma: interface block structure, kernel and transversality"),
]

def main():
    import sympy
    results = []
    ok = True
    for name, scope in SCRIPTS:
        path = HERE / name
        if not path.exists():
            results.append({"script": name, "scope": scope, "status": "missing"}); ok = False; continue
        t0 = time.time()
        r = subprocess.run([sys.executable, str(path)], cwd=HERE, capture_output=True, text=True, encoding="utf-8", errors="replace")
        dt = time.time() - t0
        passed = r.returncode == 0
        ok &= passed
        results.append({"script": name, "scope": scope, "returncode": r.returncode, "passed": passed,
                        "seconds": round(dt, 1), "stderr_tail": r.stderr[-400:] if not passed else ""})
        print(f"{'PASS' if passed else 'FAIL'}  {dt:6.1f}s  {name}")
    summary = {"all_passed": ok, "python": platform.python_version(), "sympy": sympy.__version__,
               "platform": platform.platform(), "results": results}
    (HERE / "all_certificates_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("ALL PASSED" if ok else "SOME CERTIFICATES FAILED")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
