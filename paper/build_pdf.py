import asyncio, sys
from playwright.async_api import async_playwright

CSS = """
@page { size: A4; }
body{font-family:'Times New Roman',Times,Georgia,serif;margin:0;line-height:1.48;
     font-size:10.8pt;color:#000;text-align:justify;hyphens:auto}
h1{font-size:17pt;line-height:1.3;margin:0 0 10px;font-weight:700;color:#000;text-align:left}
h2{font-size:12.5pt;margin:20px 0 6px;font-weight:700;color:#000;text-align:left;
   border:none;padding:0}
h3{font-size:11pt;margin:15px 0 4px;font-weight:700;font-style:italic;color:#000;text-align:left}
h1,h2,h3{font-family:'Times New Roman',Times,Georgia,serif}
p{margin:0 0 7px;text-indent:0}
table{border-collapse:collapse;width:100%;font-size:8.8pt;margin:9px 0;text-align:left}
th,td{border:0;border-bottom:0.5pt solid #777;padding:3.5px 6px;text-align:left;
      vertical-align:top;hyphens:none}
thead th{border-top:1pt solid #000;border-bottom:0.75pt solid #000;font-weight:700}
tbody tr:last-child td{border-bottom:1pt solid #000}
img{max-width:100%;display:block;margin:12px auto 3px;page-break-inside:avoid}
hr{display:none}
ul,ol{margin:0 0 7px;padding-left:20px}
li{margin:2px 0}
blockquote{border:none;margin:6px 0 6px 18px;padding:0;color:#000;font-style:italic}
code,pre{font-family:'Courier New',monospace;font-size:8.6pt}
pre{background:#f4f4f4;padding:7px;border-radius:2px;overflow-x:auto;text-align:left}
sup{font-size:0.72em}
"""

async def main(src, out):
    async with async_playwright() as p:
        b = await p.chromium.launch(); pg = await b.new_page()
        await pg.goto(f"file://{src}")
        await pg.add_style_tag(content=CSS)
        await pg.wait_for_timeout(3000)
        await pg.pdf(path=out, format="A4",
                     margin={"top":"20mm","bottom":"20mm","left":"22mm","right":"22mm"},
                     print_background=True, display_header_footer=True,
                     header_template="<div></div>",
                     footer_template=("<div style='width:100%;font-size:8.5pt;color:#000;"
                                      "text-align:center;font-family:Times New Roman,serif'>"
                                      "<span class='pageNumber'></span></div>"))
        await b.close()

asyncio.run(main(sys.argv[1], sys.argv[2]))
