import sys, argparse

def main():
    p = argparse.ArgumentParser(description="Run HistoQC on one or more slides (thin wrapper).")
    p.add_argument("input", nargs="+", help="Slide(s) or glob patterns (e.g. *.svs)")
    p.add_argument("-o", "--outdir", default=None, help="Output directory (default HistoQC timestamped dir)")
    p.add_argument("-n", "--nprocs", type=int, default=4, help="Number of processes")
    args = p.parse_args()

    print(f"[PySlyde-HistoQC] Starting… nprocs={args.nprocs}, outdir={args.outdir or '(default)'}")
    print(f"[PySlyde-HistoQC] Inputs: {', '.join(args.input)}")

    import histoqc.__main__ as hq  # import after parsing for clearer errors

    # Build argv for HistoQC
    sys.argv = ["histoqc", "-n", str(args.nprocs)]
    if args.outdir:
        sys.argv += ["-o", args.outdir]
    sys.argv += args.input

    print(f"[PySlyde-HistoQC] Delegating to: {' '.join(sys.argv)}")
    rc = hq.main()
    print("[PySlyde-HistoQC] Done ✅" if rc == 0 else f"[PySlyde-HistoQC] Finished with code {rc} ❗ Check error.log.")
    raise SystemExit(rc)

if __name__ == "__main__":
    main()
