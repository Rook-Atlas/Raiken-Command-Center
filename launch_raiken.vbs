' Hidden launcher for Raiken — runs pythonw.exe with no console window.
' Stdout/stderr are captured in logs/raiken.log (main.py handles the redirect).
Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")
scriptDir = FSO.GetParentFolderName(WScript.ScriptFullName)
cmd = """" & scriptDir & "\.venv\Scripts\pythonw.exe"" """ & scriptDir & "\main.py"""
' Second arg: 0 = hidden. Third arg: False = don't wait.
WshShell.Run cmd, 0, False
