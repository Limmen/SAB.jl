using Pkg
Pkg.activate(".")
using SAB

N = 2
b1 = AptGame.b1(N)
file_path = "V.csv"
alpha_vectors = OsPosgFile.parse_value_function(file_path)
value = OsPosgUtil.calculate_value(alpha_vectors, b1)
println("Value: $value")