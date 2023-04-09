import data

def test_addition():
  a = data.Value(3.0)
  b = a + a

  b.grad = 1.0
  b.backprop()

  assert b.grad == 1.0
  assert b.data == 6.0

  assert a.grad == 2.0
  assert a.data == 3.0

def test_multiplication():
  a = data.Value(2.0)
  b = data.Value(4.0)
  c = a * b

  c.grad = 1.0
  c.backprop()

  assert c.data == 8.0
  assert c.grad == 1.0

  assert a.grad == 4.0
  assert b.grad == 2.0

def test_complex():
  a = data.Value(-2.0)
  b = data.Value(3.0)

  c = a + b # 1 
  d = a * b # -6

  e = c * d

  e.grad = 1.0
  e.backprop()

  assert e.data == -6.0
  assert e.grad == 1.0

  assert c.data == 1.0
  assert c.grad == -6.0

  assert d.data == -6.0
  assert d.grad == 1.0

  assert a.data == -2.0
  assert a.grad == -3.0

  assert b.data == 3.0
  assert b.grad == -8.0


def tests():
  print("Running tests")
  test_addition()
  test_multiplication()
  test_complex()
  print("Tests run successfully")

def main():
  tests()

if __name__ == "__main__":
    main()
