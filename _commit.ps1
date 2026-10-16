Set-Location -LiteralPath 'D:\Desktop\算法代替仿真'
Remove-Item -LiteralPath 'D:\Desktop\算法代替仿真\_edit.ps1' -Force
git add -A
git -c core.quotepath=false status --porcelain | Measure-Object -Line | ForEach-Object { Write-Output ("staged entries: " + $_.Lines) }
git commit -m "清理工作区：删除一次性脚本、克隆仓库、过期图与旧版本文档（内容均保留在历史中）" | Select-Object -Last 3
git log --oneline -3