$w = New-Object -ComObject WScript.Shell
$s = $w.CreateShortcut('D:\Projects\corecast\CoreCast.lnk')
$s.TargetPath = 'D:\Projects\corecast\.venv\Scripts\pythonw.exe'
$s.Arguments = '"D:\Projects\corecast\gui.py"'
$s.WorkingDirectory = 'D:\Projects\corecast'
$s.IconLocation = 'D:\Projects\corecast\assets\corecast.ico,0'
$s.Description = 'CoreCast - video to summary'
$s.Save()
Write-Output 'shortcut created'
