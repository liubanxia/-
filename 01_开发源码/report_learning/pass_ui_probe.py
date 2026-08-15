from pywinauto import Desktop


def list_windows():
    desktop = Desktop(backend="uia")

    print("=== Phoenix Windows窗口探测 ===")

    for w in desktop.windows():
        try:
            title = w.window_text().strip()
            if not title:
                continue

            info = w.element_info

            print()
            print("标题：", title)
            print("PID：", info.process_id)
            print("类型：", info.control_type)
            print("Class：", info.class_name)

        except Exception:
            continue


if __name__ == "__main__":
    list_windows()


def find_edit_controls(keyword):
    import re

    desktop = Desktop(backend="uia")

    win = desktop.window(
        title_re=".*" + re.escape(keyword) + ".*"
    )

    win.wait("exists", timeout=5)

    print("=== 编辑控件候选 ===")
    print("目标窗口：", win.window_text())

    count = 0

    for c in win.descendants():
        try:
            info = c.element_info

            if info.control_type not in {
                "Edit",
                "Document",
            }:
                continue

            count += 1

            print()
            print("编号：", count)
            print("类型：", info.control_type)
            print("Class：", info.class_name)
            print("AutomationID：", info.automation_id)

        except Exception:
            continue

    print()
    print("候选数量：", count)


def find_edit_controls_win32(keyword):
    import re

    desktop = Desktop(
        backend="win32"
    )

    win = desktop.window(
        title_re=".*" + re.escape(keyword) + ".*"
    )

    win.wait(
        "exists",
        timeout=5
    )

    print("=== Win32编辑控件候选 ===")
    print(
        "目标窗口：",
        win.window_text()
    )

    count = 0

    for c in win.descendants():
        try:
            cls = c.class_name()

            if cls not in {
                "Edit",
                "RichEdit",
                "RichEdit20A",
                "RichEdit20W",
                "RICHEDIT50W",
            }:
                continue

            count += 1

            print()
            print(
                "编号：",
                count
            )
            print(
                "Class：",
                cls
            )
            print(
                "Handle：",
                hex(c.handle)
            )

        except Exception:
            continue

    print()
    print(
        "候选数量：",
        count
    )


def auto_probe_edit_controls(keyword):
    print("=== UIA后端 ===")
    try:
        find_edit_controls(keyword)
    except Exception as exc:
        print("UIA失败：", exc)

    print()
    print("=== Win32后端 ===")
    try:
        find_edit_controls_win32(keyword)
    except Exception as exc:
        print("Win32失败：", exc)


