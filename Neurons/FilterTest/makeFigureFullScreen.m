function makeFigureFullScreen(fig)

    s = get(0, 'ScreenSize');
    set(fig, 'Position', [0 0 s(3) s(4)]);