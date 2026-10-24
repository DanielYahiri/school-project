import ast
import json
import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_DIR / "occupation du sol predict.ipynb"
OUTPUT_DIR = PROJECT_DIR / "terminal_outputs"


def print_result(value):
    if value is None:
        return

    if hasattr(value, "to_string"):
        print(value.to_string())
    else:
        print(value)


def display(*values):
    for value in values:
        print_result(value)


def save_open_figures(plt, figure_count):
    for number in plt.get_fignums():
        figure = plt.figure(number)
        figure_count[0] += 1
        OUTPUT_DIR.mkdir(exist_ok=True)
        filename = OUTPUT_DIR / f"figure_{figure_count[0]:02d}.png"
        figure.savefig(filename, bbox_inches="tight", dpi=150)
        print(f"Graphique enregistré : {filename.relative_to(PROJECT_DIR)}")
        plt.close(figure)


def main():
    os.chdir(PROJECT_DIR)
    os.environ["MPLBACKEND"] = "Agg"

    with NOTEBOOK_PATH.open(encoding="utf-8") as notebook_file:
        notebook = json.load(notebook_file)

    namespace = {
        "__name__": "__main__",
        "__file__": str(NOTEBOOK_PATH),
        "display": display,
        "print_result": print_result,
    }
    plot_saving_configured = False
    figure_count = [0]

    for cell_number, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        if not source.strip():
            continue

        print(f"\n--- Cellule {cell_number} ---")
        tree = ast.parse(source, filename=f"{NOTEBOOK_PATH} (cellule {cell_number})")
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            final_expression = tree.body.pop()
            result_name = f"__cell_result_{cell_number}"
            tree.body.append(
                ast.Assign(
                    targets=[ast.Name(id=result_name, ctx=ast.Store())],
                    value=final_expression.value,
                )
            )
            tree.body.append(
                ast.Expr(
                    value=ast.Call(
                        func=ast.Name(id="print_result", ctx=ast.Load()),
                        args=[ast.Name(id=result_name, ctx=ast.Load())],
                        keywords=[],
                    )
                )
            )

        ast.fix_missing_locations(tree)
        exec(compile(tree, f"{NOTEBOOK_PATH} (cellule {cell_number})", "exec"), namespace)

        if not plot_saving_configured and "plt" in namespace:
            namespace["plt"].show = lambda *args, **kwargs: save_open_figures(
                namespace["plt"], figure_count
            )
            plot_saving_configured = True

    print("\nExécution du notebook terminée.")


if __name__ == "__main__":
    main()
