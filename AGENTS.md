@CLAUDE.md

# 环境配置
python环境在.venv.
你必须进入.venv才能运行pyhon

# 依赖增加
你所有确定增加的依赖必须在pyproject.toml中增加
你可以临时性使用uv pip install来增加依赖，但务必确保更新pyproject.toml

# 浏览器测试
你需要使用playwright来测试浏览器端的功能
你需要使用zai-mcp-server来分析截屏，但应注意Playwright/playwright_screenshot的输出中，文件地址在响应中形如：
```
Screenshot saved to: Downloads\welcome-page-2026-05-28T14-47-26-213Z.png
```
这里的Downloads对应到`C:\Users\sjmin\Downloads`，后面就是文件名。