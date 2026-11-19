import sqlite3

con = sqlite3.connect('storage/apac_face.db')
cur = con.cursor()
cur.execute("UPDATE datasets SET status = 'COMPLETED' WHERE id = 8")
con.commit()
print("Updated dataset 8 status to COMPLETED")
con.close()

