import sqlite3

conn = sqlite3.connect("storage/apac_face.db")
c = conn.cursor()
c.execute("UPDATE datasets SET status = 'COMPLETED' WHERE id = 8")
conn.commit()
conn.close()
print("Updated Dataset 8 status to COMPLETED!")
