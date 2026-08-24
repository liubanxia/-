from __future__ import annotations

import types
import unittest

from phoenix_knowledge.gui_bootstrap import (
    GUIBootstrapError,
    GUI_CONTRACT_VERSION,
    GUI_INSTALL_ORDER,
    install_gui_stack,
)


class _Window:
    pass


class GUIBootstrapContractTests(unittest.TestCase):
    def test_install_order_is_single_and_deterministic(self):
        gui_module = types.SimpleNamespace(WorkbenchWindow=_Window)
        calls: list[str] = []

        def loader(fullname: str):
            short = fullname.rsplit(".", 1)[-1]

            def install(_gui_module):
                calls.append(short)

            return types.SimpleNamespace(install=install)

        applied = install_gui_stack(
            gui_module,
            strict=True,
            module_loader=loader,
        )
        self.assertEqual(applied, GUI_INSTALL_ORDER)
        self.assertEqual(tuple(calls), GUI_INSTALL_ORDER)
        self.assertEqual(
            gui_module.WorkbenchWindow.__phoenix_gui_contract__,
            GUI_CONTRACT_VERSION,
        )
        self.assertEqual(
            gui_module.WorkbenchWindow.__phoenix_gui_install_order__,
            GUI_INSTALL_ORDER,
        )
        self.assertEqual(
            gui_module.WorkbenchWindow.__phoenix_gui_bootstrap_failures__,
            (),
        )

    def test_strict_bootstrap_rejects_partial_gui(self):
        gui_module = types.SimpleNamespace(WorkbenchWindow=type("Window2", (), {}))

        def loader(fullname: str):
            short = fullname.rsplit(".", 1)[-1]
            if short == "document_gui":
                raise RuntimeError("simulated installer failure")
            return types.SimpleNamespace(install=lambda _gui_module: None)

        with self.assertRaises(GUIBootstrapError):
            install_gui_stack(
                gui_module,
                strict=True,
                module_loader=loader,
            )


if __name__ == "__main__":
    unittest.main()
