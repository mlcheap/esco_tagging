createdb -U postgres mydb1
psql -U postgres mydb1 < ./data/mydb1.sql
createdb -U postgres mydb2
psql -U postgres mydb2 < ./data/mydb2.sql
