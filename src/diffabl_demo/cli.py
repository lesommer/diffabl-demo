"""Command-line interface for diffabl-demo."""

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="diffabl-demo",
        description="Differentiable simplified atmospheric boundary layer model",
    )
    sub = parser.add_subparsers(dest="command")

    p_andren = sub.add_parser("andren94", help="Andren 1994 Ekman spiral (Fig. 4)")
    p_andren.add_argument("--steps", type=int, default=1670, help="Number of steps")
    p_andren.add_argument("--params", type=str, default="cbr", choices=["cbr", "cch"])

    p_cuxart = sub.add_parser("cuxart05", help="Cuxart 2005 convective ABL (Fig. 5)")
    p_cuxart.add_argument("--steps", type=int, default=3240, help="Number of steps")
    p_cuxart.add_argument("--params", type=str, default="cch", choices=["cbr", "cch"])

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "andren94":
        from diffabl_demo.demos import andren94
        from diffabl_demo.state import cbr_params, cch_params
        params = cbr_params(dt=60.0, f=1e-4) if args.params == "cbr" else cch_params(dt=60.0, f=1e-4)
        state, grid = andren94(params, n_steps=args.steps)
        print(f"Andren94 completed: u[1]={float(state.u[1]):.3f}, v[1]={float(state.v[1]):.3f}, pblh={float(state.pblh[0]):.1f}")

    elif args.command == "cuxart05":
        from diffabl_demo.demos import cuxart05
        from diffabl_demo.state import cbr_params, cch_params
        params = cch_params(dt=10.0, f=1.39e-4) if args.params == "cch" else cbr_params(dt=10.0, f=1.39e-4)
        state, grid = cuxart05(params, n_steps=args.steps)
        print(f"Cuxart05 completed: u[1]={float(state.u[1]):.3f}, theta[1]={float(state.theta[1]):.3f}, pblh={float(state.pblh[0]):.1f}")


if __name__ == "__main__":
    main()
