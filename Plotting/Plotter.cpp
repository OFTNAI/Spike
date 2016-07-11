#include "Plotter.h"

// #include "../SpikeAnalyser/SpikeAnalyser.h"

// Plotter Constructor
Plotter::Plotter() {
	
}


// Plotter Destructor
Plotter::~Plotter() {

}



void Plotter::plot_single_cell_information_analysis(SpikeAnalyser * spike_analyser_for_untrained_network, SpikeAnalyser * spike_analyser_for_trained_network) {
	mglGraph *gr = new mglGraph();
	gr->SetFontSize(0.8);
	gr->SetSize(2000,2000);

	mglData dataUNTRAINED(spike_analyser_for_untrained_network->number_of_neurons_in_group);
	mglData dataTRAINED(spike_analyser_for_trained_network->number_of_neurons_in_group);
	mglData neuron_indices_zeroed(spike_analyser_for_trained_network->number_of_neurons_in_group);

	for (int neuron_index_zeroed = 0; neuron_index_zeroed < spike_analyser_for_untrained_network->number_of_neurons_in_group; neuron_index_zeroed++) {
		printf("spike_analyser_for_untrained_network->descending_maximum_information_score_for_each_neuron[%d]: %f\n", neuron_index_zeroed, spike_analyser_for_untrained_network->descending_maximum_information_score_for_each_neuron[neuron_index_zeroed]);
		printf("spike_analyser_for_trained_network->descending_maximum_information_score_for_each_neuron[%d]: %f\n", neuron_index_zeroed, spike_analyser_for_trained_network->descending_maximum_information_score_for_each_neuron[neuron_index_zeroed]);
		dataUNTRAINED.a[neuron_index_zeroed] = spike_analyser_for_untrained_network->descending_maximum_information_score_for_each_neuron[neuron_index_zeroed];
		dataTRAINED.a[neuron_index_zeroed] = spike_analyser_for_trained_network->descending_maximum_information_score_for_each_neuron[neuron_index_zeroed];
		neuron_indices_zeroed.a[neuron_index_zeroed] = neuron_index_zeroed;
	}

	gr->SubPlot(1,1,0,"<_");
	gr->Title("Single Cell Information Analysis");
	gr->SetRanges(0,spike_analyser_for_untrained_network->number_of_neurons_in_group,0,spike_analyser_for_untrained_network->maximum_possible_information_score);	gr->Axis();

	gr->Plot(dataUNTRAINED);
	gr->Plot(dataTRAINED);

	// gr->Plot(dataUNTRAINED, "b.", "legend 'dataUNTRAINED'");
	gr->Plot(dataUNTRAINED, dataTRAINED, "b., r.");
	// gr->Plot(dataTRAINED, neuron_indices_zeroed, "r.", "legend 'dataTRAINED'");
	gr->Legend(2);

	// // gr->Bifurcation(0.005,"x*y*(1-y)","r");

	 gr->WriteFrame("single_cell_information_analysis.png");
}


void Plotter::multiple_subplots_test() {

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

