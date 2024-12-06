TOKEN="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"

git clone https://${TOKEN}@github.com/lapunik/test_cluster.git
cd test_cluster

python3 main.py

# rm -rf new_file.py
# touch new_file_out.py
# touch new_file_solved.py
# echo "print('Please')" >> new_file.py
# cat main.py >> new_file.py


git config user.email "vojtechlapunik@gmail.com"
git config user.name "Vojtech Lapunik"
git add .
git commit -m "Automated commit with results from cluster"
git push https://${TOKEN}@github.com/lapunik/test_cluster.git
