This is a job-description and resume anayzer project which takes input of job desctiptions from recruiters and input from candidates of their resume , analyzes the text and then based on similarity it matches the resume to the job description for easily enhancing and making ease of experience for candidates and recruiters.
UPDATE log:
11-JUNE-2025
9:45:PM : PROGRAMMED PDF_parser.py file , it is extracting the text from pdf file and storing it in a text file only if the program is runned from vs code terminal .
when fast api is run it shows the extracted file in browser in json format

TO RUN PROGRAM :

1. open vs code terminal and do git clone
2. open a new terminal and write :
   python -m venv virtualenviornment
3. activate the virtual environment by writing :
   cd .\AI-Powered-Job-Assistant\virtualenviornment\Scripts\
4. > > ./activate

now cut the requirements.txt file from the root directory and paste it in the Scripts folder inside virtualenviornment

5. > > pip install -r requirements.txt
   > > open a new terminal
6. > > cd .\AI-Powered-Job-Assistant\
cd pytAI-Powered-Job-Assistant\backend\app
7. > > uvicorn app.main:app --reload

Remember to keep atleast one pdf file inside utils "AI-Powered-Job-Assistant\backend\utils"
folder and change the name in main.py file to match the name of the pdf file you have in utils folder.

now open the local host 8000 on browser and you will see the extracted text in json format

UPTADTE LOG : 30-june-2025
added Tfidf_analyzer.py it implements tdidf and gives the probabiity of each words

to run the project follow the same steps as above and run the following command in the terminal :
cd C:\Users\nipun\OneDrive\Attachments\Desktop\resume_analyzer\AI-Powered-Job-Assistant\backend\app
and then : uvicorn app.main:app --reload

first of all you need a pdf file in the AI-Powered-Job-Assistant\backend\app :

==>

> > curl.exe -X POST -F "file=@your_file_PATH_dont_remove@" http://127.0.0.1:8000/analyze-resume/
> > \*\*dont remove the @ just stick your file path

run the above command in vs code and you get the text

for postman paste the above command but
in body tak select key as file then again select File and then in value column upload the file that you have with te same name as you gave in the curl command command

you must get alot of text incuding tf-idf values of the words.

to run this test in postman
upload the file it must show there with the same name. 
and you shall see the text in response.