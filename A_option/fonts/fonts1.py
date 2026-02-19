import matplotlib.font_manager as fm

# 列出所有可用字体（按名称排序，方便查找）
font_list = [f.name for f in fm.fontManager.ttflist]
font_list.sort()

# 打印前20个字体（也可以打印全部：print(font_list)）
print("当前环境支持的字体（前20个）：")
for font in font_list:
    print(font)

# 快速搜索是否有常用无衬线字体（比如Arial、DejaVu Sans）
print("\n=== 查找常用字体 ===")
target_fonts = ["Arial", "DejaVu Sans", "SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei"]
for font in target_fonts:
    if font in font_list:
        print(f"✅ 找到字体：{font}")
    else:
        print(f"❌ 未找到字体：{font}")