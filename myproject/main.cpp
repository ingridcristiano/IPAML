// include aia and ucas utility functions
#include "ipaConfig.h"
#include "ucasConfig.h"
#include <fstream>
// include my project functions
#include "functions.h"

int main()
{
	try
	{
		// EXAMPLE 1: load and show an RGB image
		cv::Mat imgRGB = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/gemini.png", cv::IMREAD_UNCHANGED);
		//                    /\
		//                    || like MATLAB! We just need to provide the image full path
		//                    || cv::IMREAD_UNCHANGED means "load the image as it is, without grayscale-to-color
		//                    || conversion (default) and 16-to-8-bit conversion (default). Not a problem in this case,
		//                    || but it can be in EXAMPLE 2 and 3!
		// check if image data was successfully loaded. If not, we throw an error exception
		if (!imgRGB.data)
			throw ipa::error("Cannot load image");
		// print some image characteristics (X-Y dimensions, number of channels, bitdepth)
		printf("Image loaded: dims = %d x %d, channels = %d, bitdepth = %d\n",
			imgRGB.rows, imgRGB.cols, imgRGB.channels(), ipa::bitdepth(imgRGB.depth()));
		//ipa::imshow("An RGB image", imgRGB);
		//     
		//     || ipa::imshow is a slightly modified version of cv::imshow (see EXAMPLE 3 for the additional arguments)
		// EXAMPLE 2: load and show a grayscale image (8 bits per pixel)
		//cv::Mat imgGray8 = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/gemini.png", cv::IMREAD_UNCHANGED);
		cv::Mat imgGray8 = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/gemini.png", cv::IMREAD_GRAYSCALE); 
		if(!imgGray8.data)
			throw ipa::error("Cannot load image");
		printf("Image loaded: dims = %d x %d, channels = %d, bitdepth = %d\n",
			imgGray8.rows, imgGray8.cols, imgGray8.channels(), ipa::bitdepth(imgGray8.depth()));
		//ipa::imshow("An 8-bit grayscale image", imgGray8);


		// EXAMPLE 3: load and show a grayscale image (16 bits per pixel)
		cv::Mat imgGray16 = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/gemini.png", cv::IMREAD_UNCHANGED);
		if(!imgGray16.data)
			throw ipa::error("Cannot load image");
		printf("Image loaded: dims = %d x %d, channels = %d, bitdepth = %d\n",
			imgGray16.rows, imgGray16.cols, imgGray16.channels(), ipa::bitdepth(imgGray16.depth()));
		//ipa::imshow("An 16-bit grayscale image", imgGray16, true, 0.3f);
		                                                 //  /\    /\
		                                                 //  ||    || scale factor (mammograms are too big to fit in the screeen!)
		                                                 //  || wait for the user to press a key before closing the window

		// 1. Conversione in float (necessaria per calcoli matematici precisi)
		cv::Mat imgFloat;
		imgGray8.convertTo(imgFloat, CV_32F, 1.0 / 255.0);

		// 2. Applicazione del logaritmo per esaltare le sorgenti deboli
		cv::log(imgFloat + 1.0f, imgFloat);

		// 3. Normalizzazione e ritorno al formato 8-bit per la visualizzazione
		cv::normalize(imgFloat, imgGray8, 0, 255, cv::NORM_MINMAX, CV_8U);

		// 4. Mostra il risultato del pre-processing
		//ipa::imshow("Immagine Post-Logaritmo", imgGray8);

		// 5. Creazione dell'elemento strutturante (un cerchio di raggio 15)
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));

		// 6. Applicazione del filtro Top-Hat (White Top-Hat)
		cv::morphologyEx(imgGray8, imgGray8, cv::MORPH_TOPHAT, kernel);

		// 7. Mostra l'immagine pulita
		ipa::imshow("Immagine Post-TopHat", imgGray8);

		// 8. Creazione della maschera binaria con l'algoritmo di Otsu
		cv::Mat imgBin;
		cv::threshold(imgGray8, imgBin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

		// 9. Mostra il risultato della segmentazione
		ipa::imshow("Maschera Binaria", imgBin);

		// 10. Estrazione delle componenti connesse (oggetti isolati)
		cv::Mat labels, stats, centroids;
		int nObjects = cv::connectedComponentsWithStats(imgBin, labels, stats, centroids);

		// 11. Stampa il numero di oggetti trovati (escludendo lo sfondo)
		printf("Numero di oggetti rilevati: %d\n", nObjects - 1);

		// 12. Creazione e apertura del file CSV
		std::ofstream csvFile("dataset_estratto.csv");

		// Scriviamo l'intestazione delle colonne (le nostre Feature)
		csvFile << "ID;Centroid_X;Centroid_Y;Area;Bounding_Width;Bounding_Height\n";

		// 13. Ciclo for per analizzare ogni singolo oggetto (partiamo da 1, perché 0 è lo sfondo nero)
		for (int i = 1; i < nObjects; i++) {

			// Estrazione coordinate del centro
			double cx = centroids.at<double>(i, 0);
			double cy = centroids.at<double>(i, 1);

			// Estrazione statistiche geometriche di base
			int area = stats.at<int>(i, cv::CC_STAT_AREA);
			int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
			int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);

			// Scrittura della riga nel CSV
			csvFile << i << ";" << cx << ";" << cy << ";" << area << ";" << width << ";" << height << "\n";
		}

		csvFile.close(); // Chiudiamo il file per salvare i dati
		printf("Estrazione completata con successo! Dati salvati in dataset_estratto.csv\n");

		// 14. Comando per aprire automaticamente il file CSV (specifico per Windows)
		system("start dataset_estratto.csv");
		//// EXAMPLE 4: do some image processing and save the result
		//// histogram equalization is just one of MANY functions available in OpenCV
		//cv::equalizeHist(imgGray8, imgGray8);
		//cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/lowcontrast.EQ.png", imgGray8);


		// EXAMPLE 5: video processing: Face Detection
		// - requires a (working) camera
		// - requires a Face Detector (pre-trained classifier): automatically loaded within faceRectangles()
		/* comment this if you do not want to run it / it does not work */
		//ipa::processVideoStream("", ipa::project0::faceRectangles);
		//                      /\                       /\
		//                      || empty video input = use camera
		//                                               || function / processing to be applied to each frame of the video sequence

		return EXIT_SUCCESS;
	}
	catch (ipa::error &ex)
	{
		std::cout << "EXCEPTION thrown by " << ex.getSource() << "source :\n\t|=> " << ex.what() << std::endl;
	}
	catch (ucas::Error &ex)
	{
		std::cout << "EXCEPTION thrown by unknown source :\n\t|=> " << ex.what() << std::endl;
	}
}

