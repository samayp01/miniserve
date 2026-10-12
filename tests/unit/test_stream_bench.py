from benchmarks.stream_bench import render, thirds


def test_render_draws_a_different_frame_each_time():
    assert len({render(0, i) for i in range(5)}) == 5
    assert render(0, 3) != render(1, 3)


def test_thirds_compares_the_start_and_end_of_a_run():
    samples = [{"x": v} for v in (1, 1, 1, 5, 5, 5, 9, 9, 9)]
    assert thirds(samples, "x") == (1, 9, 9)
