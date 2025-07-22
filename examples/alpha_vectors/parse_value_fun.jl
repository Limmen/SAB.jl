using Pkg
Pkg.activate(".")
using SAB

N = 1
b1 = AptGame.b1(N)
file_path = "V.csv"
alpha_vectors = OsPosgFile.parse_value_function(file_path)
value = OsPosgUtil.calculate_value(alpha_vectors, b1)
println("Value: $value")

for b1 in 0:0.01:1
    b0 = 1-b1
    b = [b0, b1]
    value = OsPosgUtil.calculate_value(alpha_vectors, b)
    println("$b1 $value")
end