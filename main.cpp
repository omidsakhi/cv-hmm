/*
 *      C++ Sample Code using Hidden Markov Model for OpenCV (CvHMM).
 *
 * Copyright (c) 2012 Omid B. Sakhi
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <iostream>
#include <time.h>

#include <opencv2/core/core.hpp>
#include "CvHMM.h"

int main()
{
    std::cout << "First we define Transition, Emission\nand Initial Probabilities of the model\n\n";
    double TRANSdata[] = {0.5 , 0.5 , 0.0,
                          0.0 , 0.7 , 0.3,
                          0.0 , 0.0 , 1.0};
    cv::Mat TRANS = cv::Mat(3,3,CV_64F,TRANSdata).clone();
    double EMISdata[] = {2.0/4.0 , 2.0/4.0 , 0.0/4.0 , 0.0/4.0 ,
                         0.0/4.0 , 2.0/4.0 , 2.0/4.0 , 0.0/4.0 ,
                         0.0/4.0 , 0.0/4.0 , 2.0/4.0 , 2.0/4.0 };
    cv::Mat EMIS = cv::Mat(3,4,CV_64F,EMISdata).clone();
    double INITdata[] = {1.0  , 0.0 , 0.0};
    cv::Mat INIT = cv::Mat(1,3,CV_64F,INITdata).clone();
    CvHMM hmm;
    hmm.printModel(TRANS,EMIS,INIT);
    //----------------------------------------------------------------------------------
    std::cout << "\nAs an example, we generate 25 sequences each with 20 observations\nper sequence using the defined Markov model\n";
    srand ((unsigned int) time(NULL) );
    cv::Mat seq,states;
    hmm.generate(20,25,TRANS,EMIS,INIT,seq,states);
    std::cout << "\nGenerated Sequences:\n";
    for (int i=0;i<seq.rows;i++)
    {
        std::cout << i << ": ";
        for (int j=0;j<seq.cols;j++)
            std::cout << seq.at<int>(i,j);
        std::cout << "\n";
    }
    std::cout << "\nGenerated States:\n";
    for (int i=0;i<seq.rows;i++)
    {
        std::cout << i << ": ";
        for (int j=0;j<seq.cols;j++)
            std::cout << states.at<int>(i,j);
        std::cout << "\n";
    }
    std::cout << "\n";
    //----------------------------------------------------------------------------------
    std::cout << "\nProblem 1: Given the observation sequence and a model,\n";
    std::cout << "how do we efficiently compute P(O|Y), the probability of\n";
    std::cout << "the observation sequence, given the model?\n";
    std::cout << "Example: To demonstrate this we estimate the probabilities\n";
    std::cout << "for all sequences, given the defined model above.\n";
    cv::Mat pstates,forward,backward;
    double logpseq;
    std::cout << "\n";
    for (int i=0;i<seq.rows;i++)
    {
        hmm.decode(seq.row(i),TRANS,EMIS,INIT,logpseq,pstates,forward,backward);
        std::cout << "logpseq" << i << " " << logpseq << "\n";
    }
    std::cout << "\n";
    //----------------------------------------------------------------------------------
    std::cout << "\nProblem 2: Given the model and an observation sequence,\n";
    std::cout << "how do we find an optimal state sequence for the underlying\n";
    std::cout << "Markov Process? One answer is by using Viterbi algorithm.\n";
    std::cout << "As an example here we estimate the optimal states for all sequences\n";
    std::cout << "using Viterbi algorithm and the defined model.\n";
    cv::Mat estates;
    std::cout << "\n";
    for (int i=0;i<seq.rows;i++)
    {
        std::cout << i << ": ";
        hmm.viterbi(seq.row(i),TRANS,EMIS,INIT,estates);
        for (int i=0;i<estates.cols;i++)
            std::cout << estates.at<int>(0,i);
        std::cout << "\n";
    }
    std::cout << "\n";
    //----------------------------------------------------------------------------------
    std::cout << "\nProblem 3: Given an observation sequence O (can be several observations),\n";
    std::cout << "how do we find a model that maximizes the probability of O ?\n";
    std::cout << "The answer is by using the Baum-Welch algorithm to train a model.\n";
    std::cout << "To demonstrate this, initially we define a model by guess\n";
    std::cout << "and we estimate the parameters of the model for all the sequences\n";
    std::cout << "that we already got.\n";
    double TRGUESSdata[] = {2.0/3.0 , 1.0/3.0 , 0.0/3.0,
                            0.0/3.0 , 2.0/3.0 , 1.0/3.0,
                            0.0/3.0 , 0.0/3.0 , 3.0/3.0};
    cv::Mat TRGUESS = cv::Mat(3,3,CV_64F,TRGUESSdata).clone();
    double EMITGUESSdata[] = {1.0/4.0 , 1.0/4.0 , 1.0/4.0 , 1.0/4.0 ,
                              1.0/4.0 , 1.0/4.0 , 1.0/4.0 , 1.0/4.0 ,
                              1.0/4.0 , 1.0/4.0 , 1.0/4.0 , 1.0/4.0 };
    cv::Mat EMITGUESS = cv::Mat(3,4,CV_64F,EMITGUESSdata).clone();
    double INITGUESSdata[] = {0.6  , 0.2 , 0.2};
    cv::Mat INITGUESS = cv::Mat(1,3,CV_64F,INITGUESSdata).clone();
    hmm.train(seq,100,TRGUESS,EMITGUESS,INITGUESS);
    hmm.printModel(TRGUESS,EMITGUESS,INITGUESS);
    //----------------------------------------------------------------------------------
    std::cout << "\ndone.\n";    
    return 0;
}
