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

        // EXAMPLE 2: load and show a grayscale image (8 bits per pixel)

        //cv::Mat imgGray8 = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/gemini.png", cv::IMREAD_UNCHANGED);

        cv::Mat imgGray8 = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/gemini.png", cv::IMREAD_GRAYSCALE);

        if (!imgGray8.data)

            throw ipa::error("Cannot load image");

        printf("Image loaded: dims = %d x %d, channels = %d, bitdepth = %d\n",

            imgGray8.rows, imgGray8.cols, imgGray8.channels(), ipa::bitdepth(imgGray8.depth()));

        //ipa::imshow("An 8-bit grayscale image", imgGray8);



        // 1. Conversione in float (necessaria per calcoli matematici precisi)

        cv::Mat imgFloat;

        imgGray8.convertTo(imgFloat, CV_32F, 1.0 / 255.0);



        // 2. Applicazione del logaritmo per esaltare le sorgenti deboli

        cv::log(imgFloat + 1.0f, imgFloat);



        // 3. Normalizzazione e ritorno al formato 8-bit per la visualizzazione

        cv::normalize(imgFloat, imgGray8, 0, 255, cv::NORM_MINMAX, CV_8U);



        // 4. Mostra il risultato del pre-processing

       // ipa::imshow("Immagine Post-Logaritmo", imgGray8);



        // 5. Creazione dell'elemento strutturante (un cerchio di raggio 15)

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));



        // 6. Applicazione del filtro Top-Hat (White Top-Hat)

        cv::morphologyEx(imgGray8, imgGray8, cv::MORPH_TOPHAT, kernel);



        // 7. Mostra l'immagine pulita

       // ipa::imshow("Immagine Post-TopHat", imgGray8);



        // 8. Creazione della maschera binaria con l'algoritmo di Otsu

        cv::Mat imgBin;

        cv::threshold(imgGray8, imgBin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);



        // 9. Mostra il risultato della segmentazione

        //ipa::imshow("Maschera Binaria", imgBin);



        // 10. Estrazione delle componenti connesse (oggetti isolati)

        cv::Mat labels, stats, centroids;

        int nObjects = cv::connectedComponentsWithStats(imgBin, labels, stats, centroids);



        // 11. Stampa il numero di oggetti trovati (escludendo lo sfondo)

        printf("Numero di oggetti rilevati: %d\n", nObjects - 1);



        // 12. Creazione e apertura del file CSV

        std::ofstream csvFile("dataset_estratto.csv");



        // Scriviamo l'intestazione 

        csvFile << "ID;Centroid_X;Centroid_Y;Area;Width;Height;Compactness;Eccentricity\n";



        // 13. Ciclo for per analizzare ogni singolo oggetto

        for (int i = 1; i < nObjects; i++) {



            int area = stats.at<int>(i, cv::CC_STAT_AREA);



            // FILTRO RUMORE: Analizziamo solo gli oggetti con area maggiore di 10 pixel

            if (area > 10) {

                // Estrazione coordinate del centro

                double cx = centroids.at<double>(i, 0);

                double cy = centroids.at<double>(i, 1);



                // Estrazione dimensioni bounding box

                int width = stats.at<int>(i, cv::CC_STAT_WIDTH);

                int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);



                // CALCOLO COMPATTEZZA: Area / (Larghezza * Altezza)

               double compactness = (double)area / (width * height);

			   //calcolo eccentricity 

               
               
               
          

			   //scrittura su file CSV
               csvFile << i << ";" << cx << ";" << cy << ";" << area << ";" << width << ";" << height << ";" << compactness << "\n";
               
              
            }

        }



        csvFile.close(); // Chiudiamo il file per salvare i dati

        printf("Estrazione completata con successo! Dati salvati in dataset_estratto.csv\n");


        // 14. Comando per aprire automaticamente il file CSV (specifico per Windows)

        system("start dataset_estratto.csv");





        return EXIT_SUCCESS;

    }

    catch (ipa::error& ex)

    {

        std::cout << "EXCEPTION thrown by " << ex.getSource() << "source :\n\t|=> " << ex.what() << std::endl;

    }

    catch (ucas::Error& ex)

    {

        std::cout << "EXCEPTION thrown by unknown source :\n\t|=> " << ex.what() << std::endl;

    }

}