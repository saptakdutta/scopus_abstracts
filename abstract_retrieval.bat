@echo off
call C:\Users\duttas\AppData\Local\miniconda3\condabin\activate
call activate scopus_abstract_mining

@echo on
python abstract_retrieval_by_uniname.py

@echo off
call conda deactivate
call conda deactivate

pause
goto :eof
