canvasSize = 128;
I = zeros(canvasSize,canvasSize)+128;
imageSize = 50;

xbegin = (canvasSize-imageSize)/2;
ybegin = (canvasSize-imageSize)/2;

pos = [xbegin ybegin imageSize imageSize];
I = rectangle('Position',pos,'Curvature',[1 1])
