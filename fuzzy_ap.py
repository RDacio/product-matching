from fuzzywuzzy import fuzz
from fuzzywuzzy import process

#client description
client_desc = "TESOURA FINA PCT BICUDA 115MM"

#db descriptions
db_o_descriptions = ["TESOURA IRIS CURVA 115MM ", "TESOURA FINA P.RCT.BICUDA 115MM", "TESOURA IRIS RETA 130MM"]

#calculate similarity
for db_o_description in db_o_descriptions:
    print(fuzz.ratio(client_desc, db_o_description))

#scores
scores = [fuzz.ratio(client_desc, db_o_description) for db_o_desc in db_o_descriptions]

#match
match = process.extractOne(client_desc, db_o_descriptions)

print(f"Client Description: {client_desc}")
print(f"Best Match in DB: {match[0]} (Score: {match[1]})")

print(process.extract(client_desc, db_o_descriptions, limit=3))