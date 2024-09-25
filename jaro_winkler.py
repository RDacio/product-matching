from jaro import jaro_winkler_metric

def jaro_distance(s1, s2):
    return jaro_winkler_metric(s1, s2)

s1 = "TESOURA FINA PCT BICUDA 115MM"
s2 = "TESOURA FINA P.RCT.BICUDA 115MM"

distance = jaro_distance(s1, s2)
print(f"Jaro Distance: {distance:.4f}")