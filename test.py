from classes import cancer, glass, votes, iris, soybean

att = [123456, 0, 2, 4, 1, 0, 5, 3, 8, 0, 1]

learnable_classes = {
  "cancer": cancer.Cancer,
  "glass": glass.Glass,
  "votes": votes.Votes,
  "iris": iris.Iris,
  "soybean": soybean.Soybean
}

learnable_class = learnable_classes["cancer"]

x = learnable_class(att, True)

print(x)