import matplotlib.font_manager as font_manager

font_list = font_manager.findSystemFonts()
for font_path in font_list:
    font_props = font_manager.FontProperties(fname=font_path)
    font_name = font_props.get_name()
    # 筛选中文字体（以 Noto 或 文泉驿为例）
    if "Noto" in font_name or "WenQuanYi" in font_name:
        print(f"字体路径：{font_path}，字体名称：{font_name}")