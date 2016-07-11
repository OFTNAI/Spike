#include "Plotter.h"

#include "mgl2/mgl.h"

// Plotter Constructor
Plotter::Plotter() {

	
}


// Plotter Destructor
Plotter::~Plotter() {

}


void Plotter::test_plot() {

	mglGraph *gr = new mglGraph();		// class for plot drawing

    gr->SetSize(1000,1000);

    gr->SubPlot(2,2,0,"<_");	
    gr->Title("Semi-log axis");	gr->SetRanges(0.01,100,-1,1);	gr->SetFunc("lg(x)","");
	gr->Axis();	gr->Grid("xy","g");	gr->FPlot("sin(1/x)");	gr->Label('x',"x",0); gr->Label('y', "y = sin 1/x",0);
	gr->SubPlot(2,2,1,"<_");	gr->Title("Log-log axis");	gr->SetRanges(0.01,100,0.1,100);	gr->SetFunc("lg(x)","lg(y)");
	gr->Axis();	gr->Grid("!","h=");	gr->Grid();	gr->FPlot("sqrt(1+x^2)");	gr->Label('x',"x",0); gr->Label('y', "y = \\sqrt{1+x^2}",0);
	gr->SubPlot(2,2,2,"<_");	gr->Title("Minus-log axis");	gr->SetRanges(-100,-0.01,-100,-0.1);	gr->SetFunc("-lg(-x)","-lg(-y)");
	gr->Axis();	gr->FPlot("-sqrt(1+x^2)");	gr->Label('x',"x",0); gr->Label('y', "y = -\\sqrt{1+x^2}",0);
	gr->SubPlot(2,2,3,"<_");	gr->Title("Log-ticks");	gr->SetRanges(0.1,100,0,100);	gr->SetFunc("sqrt(x)","");
	gr->Axis();	gr->FPlot("x");	gr->Label('x',"x",1); gr->Label('y', "y = x",0);

    gr->WriteFrame("test_plot.png");	// save it


}

