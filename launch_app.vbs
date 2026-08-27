' FloodNet app - open the app without showing the console (uses run_app.bat)
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh  = CreateObject("WScript.Shell")
sh.CurrentDirectory = fso.GetParentFolderName(WScript.ScriptFullName)
sh.Run "run_app.bat", 0, False
Set sh = Nothing
Set fso = Nothing
