from gigatorch.weight_init import WightInitializer
from math import sqrt


def test_xavier_uniform():
    rows = 3
    columns = 2
    f_in = 10
    f_out = 5

    ws = WightInitializer().xavier_uniform(f_in, f_out, rows, columns)
    assert len(ws) == rows

    limit = sqrt(6.0 / f_in + f_out)
    for i in range(len(ws)):
        assert len(ws[i]) == columns
        for j in range(len(ws[i])):
            assert abs(ws[i][j].data) < limit


def test_xavier_normal():
    rows = 3
    columns = 2
    f_in = 10
    f_out = 5

    ws = WightInitializer().xavier_normal(f_in, f_out, rows, columns)
    assert len(ws) == rows

    limit = sqrt(2.0 / f_in + f_out)
    for i in range(len(ws)):
        assert len(ws[i]) == columns
        for j in range(len(ws[i])):
            assert abs(ws[i][j].data) < limit


def test_he_normal():
    rows = 3
    columns = 2
    f_in = 10

    ws = WightInitializer().he_normal(f_in, rows, columns)
    assert len(ws) == rows

    limit = sqrt(2.0 / f_in)
    for i in range(len(ws)):
        assert len(ws[i]) == columns
        for j in range(len(ws[i])):
            assert abs(ws[i][j].data) < limit


def test_he_uniform():
    rows = 3
    columns = 2
    f_in = 10

    limit = sqrt(6.0 / f_in)
    ws = WightInitializer().he_uniform(f_in, rows, columns)
    assert len(ws) == rows

    for i in range(len(ws)):
        assert len(ws[i]) == columns
        for j in range(len(ws[i])):
            assert abs(ws[i][j].data) < limit
