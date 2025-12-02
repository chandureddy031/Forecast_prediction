docker build -t ml-app .
docker run -p 8000:8000 ml-app
dvc remote add -d localstore C:/Users/lenovo/Desktop/project/dvc_storage for local store

dvc remote remove myremote
dvc remote add -d myremote azure://dvccontainer/dvcstore

dvc remote modify --local myremote account_name "dvcstorage01"
dvc remote modify --local myremote account_key "your-key-here"

dvc push
