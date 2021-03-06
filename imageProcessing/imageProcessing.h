// This MFC Samples source code demonstrates using MFC Microsoft Office Fluent User Interface
// (the "Fluent UI") and is provided only as referential material to supplement the
// Microsoft Foundation Classes Reference and related electronic documentation
// included with the MFC C++ library software.
// License terms to copy, use or distribute the Fluent UI are available separately.
// To learn more about our Fluent UI licensing program, please visit
// https://go.microsoft.com/fwlink/?LinkId=238214.
//
// Copyright (C) Microsoft Corporation
// All rights reserved.

// imageProcessing.h : main header file for the imageProcessing application
//
#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"       // main symbols

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

#include "opencv2/core/types_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/highgui/highgui_c.h"

#include <iostream>
#include <map>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

// CImageProcessingApp:
// See imageProcessing.cpp for the implementation of this class
//

class CImageProcessingApp : public CWinAppEx
{
public:
	CImageProcessingApp() noexcept;

	CMultiDocTemplate* m_pDocTemplate;

	int		m_iCnt_pix;

	int		m_iHessianThreshold;

protected:
	vector< Ptr<FeatureDetector> >		m_detectors;
	Ptr<FeatureDetector>				m_pDetector_current;

	vector< Ptr<DescriptorExtractor> >	m_extractors;
	Ptr<DescriptorExtractor>			m_pExtractor_current;

	vector< Ptr<DescriptorMatcher> >	m_matchers;
	Ptr<DescriptorMatcher>				m_pMatcher_current;

	BOOL	m_bOnEdit_threshold;	// flag to indicate on edit Hessian threshold

// Overrides
public:
	virtual BOOL InitInstance();
	virtual int ExitInstance();

// Implementation
	UINT  m_nAppLook;
	BOOL  m_bHiColorIcons;

	virtual void PreLoadState();
	virtual void LoadCustomState();
	virtual void SaveCustomState();

	afx_msg void	OnAppAbout ( );

	afx_msg void	OnBtnMatch ( );
	afx_msg void	OnUpdateBtnMatch ( CCmdUI *pCmdUI );

	afx_msg void	OnBtnMerge ( );
	afx_msg void	OnUpdateBtnMerge ( CCmdUI *pCmdUI );

	afx_msg void	OnBtnJoint ( );
	afx_msg void	OnUpdateBtnJoint ( CCmdUI *pCmdUI );

	afx_msg void	OnSelectMatcher ( );
	afx_msg void	OnSelectDetector ( );
	afx_msg void	OnUpdateMatcher ( CCmdUI *pCmdUI );
	afx_msg void	OnUpdateDetector ( CCmdUI *pCmdUI );


	DECLARE_MESSAGE_MAP()
	afx_msg void OnBtnPoseEstimation ( );
	afx_msg void OnUpdateBtnPoseEstimation ( CCmdUI *pCmdUI );
	afx_msg void OnUpdateEditThreshold ( CCmdUI *pCmdUI );
	afx_msg void OnEditThreshold ( );
};

extern CImageProcessingApp theApp;
