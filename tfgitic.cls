\NeedsTeXFormat{LaTeX2e}[2020/02/02]
\ProvidesClass{tfgitic}[2026/01/22 UPC Manresa Bachelor/Master thesis format v8.2]
\LoadClass[a4paper,
           11pt,
           DIV=12,
           headings=big,
           twoside=semi,
           bibliography=totoc,
           %parskip=half+,
           open=any
           ]{scrbook}[2022/10/12]
\RequirePackage[T1]{fontenc}
\RequirePackage{environ,lmodern,newverbs,amsmath,amssymb}
\RequirePackage[automark,headsepline=0.4pt]{scrlayer-scrpage}
\RequirePackage{colorprofiles}
\RequirePackage[a-3b,mathxmp]{pdfx}
\RequirePackage{tikz,xcolor,booktabs}
\RequirePackage[
style=alphabetic,
datezeros=false]{biblatex}[2022/07/12]
\RequirePackage[
binary-units=true,
per-mode=symbol,
round-mode=places,
round-precision=2,
round-pad=false]{siunitx}[>v3]
\RequirePackage[autostyle=true]{csquotes}
\usetikzlibrary{svg.path}
\hypersetup{hidelinks,bookmarksnumbered=true}

% Les opcions de la classe
\newif\if@creativecommons  \@creativecommonstrue
\DeclareOption{creativecommons}{\@creativecommonstrue}
\DeclareOption{nocreativecommons}{\@creativecommonsfalse}
\newif\if@catalanlang  \@catalanlangtrue
\DeclareOption{catalan}{\@catalanlangtrue}
\DeclareOption{english}{\@catalanlangfalse}
\newif\if@master  \@masterfalse
\DeclareOption{bachelor}{\@masterfalse}
\DeclareOption{master}{\@mastertrue}
\ProcessOptions\relax

% Opcions localitzades per a siunitx
\if@catalanlang
  \sisetup{output-decimal-marker={,}}
\else
  \sisetup{output-decimal-marker={.}}
\fi

% Opcions correctes per a babel
\if@catalanlang
\PassOptionsToPackage{english,main=catalan}{babel}
\else
\PassOptionsToPackage{main=english,catalan}{babel}
\fi
\RequirePackage{babel}


% Configuració de les cometes
\DeclareQuoteAlias{spanish/spanish}{catalan}
\MakeAutoQuote{«}{»}


%%
%% La versió de la classe
%%
\def\tfgitic@version{8.2}
\newcommand{\classversion}{\tfgitic@version}

%%
%% Marca UPC normativa
%%

% El color principal, Pantone 3005
\definecolor{upcblue}{HTML}{007AC2}
% La lletra
\DeclareFixedFont{\helvb@upc}{T1}{phv}{bx}{n}{0.8cm}
\DeclareFixedFont{\helv@upc}{T1}{phv}{m}{n}{0.8cm}
% La bola upc de 10cm
\def\bola@upc{
  \path[fill=upcblue,xscale=-1,rotate=180] svg
  {M56.523,79.877h-2.051v4.275h2.109c1.367,0,1.922-1.072,1.895-2.207
    C58.443,80.84,58.002,79.877,56.523,79.877z
    M55.223,0C24.723,0,0,24.727,0,55.225c0,30.504,24.723,55.225,55.223,55.225
    c30.506,0,55.232-24.721,55.232-55.225C110.455,24.727,85.729,0,55.223,0z
    M55.445,15.592c4.355,0,7.883,3.539,7.883,7.885
    c0,4.348-3.527,7.883-7.883,7.883c-4.352,0-7.879-3.535-7.879-7.883C47.566,19.131,51.094,15.592,55.445,15.592z
    M55.445,34.277
    c4.355,0,7.883,3.525,7.883,7.881c0,4.346-3.527,7.881-7.883,7.881c-4.352,0-7.879-3.535-7.879-7.881
    C47.566,37.803,51.094,34.277,55.445,34.277z
    M36.289,15.592c4.352,0,7.887,3.539,7.887,7.885c0,4.348-3.535,7.883-7.887,7.883
    c-4.359,0-7.885-3.535-7.885-7.883C28.404,19.131,31.93,15.592,36.289,15.592z
    M36.289,34.277c4.352,0,7.887,3.525,7.887,7.881
    c0,4.346-3.535,7.881-7.887,7.881c-4.359,0-7.885-3.535-7.885-7.881C28.404,37.803,31.93,34.277,36.289,34.277z
    M36.289,53.451
    c4.352,0,7.887,3.527,7.887,7.887c0,4.352-3.535,7.875-7.887,7.875c-4.359,0-7.885-3.523-7.885-7.875
    C28.404,56.979,31.93,53.451,36.289,53.451z
    M45.58,86.982c0,1.936-0.064,4.326-2.605,6.117c-1.67,1.172-3.631,1.422-5.408,1.422
    s-3.732-0.25-5.404-1.422c-2.547-1.791-2.617-4.182-2.617-6.117v-11.09h5.313v11.066c0.014,1.049,0.111,1.875,0.93,2.545
    c0.566,0.469,1.049,0.623,1.785,0.625c0.734-0.002,1.211-0.156,1.771-0.625c0.809-0.67,0.91-1.496,0.938-2.545V75.893h5.299
    V86.982z
    M55.445,53.451c4.355,0,7.883,3.527,7.883,7.887c0,4.352-3.527,7.875-7.883,7.875c-4.352,0-7.879-3.523-7.879-7.875
    C47.566,56.979,51.094,53.451,55.445,53.451z
    M61.07,87.201c-1.016,0.719-2.168,0.994-5.117,0.994l-1.539-0.037v5.973h-4.623
    V75.893h7.639c3.723,0,6.023,2.246,6.023,6.281C63.453,84.33,62.611,86.133,61.07,87.201z
    M80.113,81.412
    c-1.807-1.518-2.521-1.406-3.945-1.406c-2.764,0-4.688,2.098-4.688,5.092c0,3.002,1.951,5.092,4.717,5.092
    c1.494,0,2.24-0.203,3.916-1.447v4.727c-1.74,0.848-2.73,1.096-4.41,1.096c-5.508,0-9.387-3.914-9.387-9.467
    c0-5.508,3.965-9.584,9.334-9.584c1.643,0,2.715,0.268,4.463,1.121V81.412z
    M74.166,69.213c-4.367,0-7.889-3.523-7.889-7.875
    c0-4.359,3.521-7.887,7.889-7.887c4.346,0,7.881,3.527,7.881,7.887C82.047,65.689,78.512,69.213,74.166,69.213z
    M74.166,50.039
    c-4.367,0-7.889-3.535-7.889-7.881c0-4.355,3.521-7.881,7.889-7.881c4.346,0,7.881,3.525,7.881,7.881
    C82.047,46.504,78.512,50.039,74.166,50.039z
    M74.166,31.359c-4.367,0-7.889-3.535-7.889-7.883c0-4.346,3.521-7.885,7.889-7.885
    c4.346,0,7.881,3.539,7.881,7.885C82.047,27.824,78.512,31.359,74.166,31.359z};
}
% La bola upc amb les lletres centrades a sota
\def\bola@upc@text{
  \bola@upc
  \draw [color=upcblue,text centered,font=\helvupc]
      (0,-7) node {UNIVERSITAT POLITÈCNICA DE CATALUNYA};
}
% L'api de la marca UPC
\newcommand{\bolaupc}[1][1cm]{\resizebox{!}{#1}{\tikz\bola@upc;}}
\newcommand{\logoupc}[1][1cm]{%
  \resizebox{!}{#1}{%
    \tikz\bola@upc
    \draw[anchor=west,color=upcblue,font=\helvb@upc]
    (4.4,-0.8) node {UNIVERSITAT POLITÈCNICA}
    (4.4,-1.95) node {DE CATALUNYA}
    (4.4,-3.0) node[font=\helv@upc] {BARCELONA\helvb@upc{}TECH};
  }
}


%%
%% Localització i generificació
%%
% generes i números
\newif\iftfgitic@counselorm\tfgitic@counselormtrue
\newif\iftfgitic@authorm\tfgitic@authormtrue
\newif\iftfgitic@advisorm\tfgitic@advisormtrue
\newif\iftfgitic@advisors\tfgitic@advisorstrue
% l10n dels strings generificats
\if@catalanlang
\def\tfgitic@defschool{Escola Politècnica Superior d'Enginyeria de Manresa}
\def\tfgitic@defdegree{Grau d'Enginyeria en Sistemes TIC}
\def\tfgitic@submittext{Treball Final de \if@master Màster\else Grau\fi\ (\tfgitic@ects{} \acro{ects}) que presenta}
\def\tfgitic@fulfilltext{en compliment dels requisits per assolir el}
\def\tfgitic@topicsname{Àmbits}
\def\tfgitic@advisorname{%
  \iftfgitic@advisorm\iftfgitic@advisors Director\else Directors\fi\else\iftfgitic@advisors Directora\else Directores\fi\fi}
\def\tfgitic@counselorname{\iftfgitic@counselorm Tutor\else Tutora\fi}
\def\tfgitic@authorname{\iftfgitic@authorm autor\else autora\fi}
\def\tfgitic@typohow{Aquest document s'ha compost fent servir
  \LaTeX{} i la classe \texttt{tfgitic} versió \tfgitic@version{}}
\else
\def\tfgitic@defschool{Manresa School of Engineering}
\def\tfgitic@defdegree{Bachelor's Degree in ICT Systems Engineering}
\def\tfgitic@submittext{\if@master Master's Thesis\else Capstone
  Project\fi\ (\tfgitic@ects{} \acro{ects}) submitted by}
\def\tfgitic@fulfilltext{in partial fulfillment of the requirements
  for the}
\def\tfgitic@topicsname{Topics}
\def\tfgitic@advisorname{\iftfgitic@advisors Advisor\else Advisors\fi}
\def\tfgitic@counselorname{Counselor}
\def\tfgitic@authorname{author}
\def\tfgitic@typohow{This manuscript has been typeset using \LaTeX{}
  and the class \texttt{tfgitic} version \tfgitic@version{}}
\fi


%%
%% Metadades del document (per a portada i material inicial)
%% algunes permeten genere/numero i actuen en conseqüència
%%

% Mostra warnings en el document
\newcommand*{\tfgitic@warn}[1]{\color{red}\normalsize #1}
% Defineix l'autor
\renewcommand*{\author}[2][m]{%
  \def\tfgitic@author{#2}\if#1f\tfgitic@authormfalse\fi%
}
% Defineix el director (amb genere i número)
\newcommand*{\advisor}[2][m]{%
  \def\@advisor{#2}%
  \if#1f\tfgitic@advisormfalse\tfgitic@advisorstrue\fi%
  \if#1F\tfgitic@advisormfalse\tfgitic@advisorsfalse\fi%
  \if#1M\tfgitic@advisormtrue\tfgitic@advisorsfalse\fi%
}
% Defineix el tutor (si es fa a empresa)
%\newif\iftfgitic@counselorm\tfgitic@counselormtrue
\newcommand*{\counselor}[2][m]{%
  \def\tfgitic@counselor{#2}\if#1f\tfgitic@counselormfalse\fi%
}
% Defineix la companyia/empresa/institució
\newcommand*{\company}[1]{\def\tfgitic@company{#1}}
\def\tfgitic@topics{\tfgitic@warn{CAL DEFINIR DOS ÀMBITS}}
\newcommand*{\topics}[1]{\def\tfgitic@topics{#1}}
\newcommand*{\commitee}[1]{}
\def\tfgitic@location{Manresa}
\newcommand*{\location}[1]{\def\tfgitic@location{#1}}
\def\tfgitic@school{\tfgitic@defschool}
\newcommand*{\school}[1]{\def\tfgitic@school{#1}}
\def\tfgitic@degree{\tfgitic@defdegree}
\newcommand*{\degree}[1]{\def\tfgitic@degree{#1}}
\newcommand*{\ects}[1]{\def\tfgitic@ects{#1}}
\def\tfgitic@ects{24}


%%
%% Portada i pàgines singulars normatives
%%

%% Bloc direcció a la portada
\newcommand*{\tfgitic@advisorblock}{
  \ifdefined\@advisor
  \ifdefined\tfgitic@counselor
    \ifdefined\tfgitic@company
      \tfgitic@advisorname:~\@advisor~(\tfgitic@company)
      \break
      \tfgitic@counselorname:~\tfgitic@counselor
    \else
      \tfgitic@warn{SI HI HA TUTOR CAL DEFINIR INSTITUCIÓ/EMPRESA}
    \fi
  \else
    \ifdefined\tfgitic@company
      \tfgitic@warn{SI HI HA INSTITUCIÓ/EMPRESA CAL DEFINIR TUTOR}
    \else
      \tfgitic@advisorname:~\@advisor
    \fi
  \fi
  \else
    \tfgitic@warn{CAL DEFINIR DIRECTOR}
  \fi
  }


% Portada normativa del treball
\renewcommand*{\maketitle}{%
  \begin{titlepage}
    \enlargethispage{20ex}
    \begin{center}
      \logoupc[1.5cm]
      \par\bigskip
      \textsf{\color{upcblue}\LARGE \tfgitic@school}
    \end{center}
    \vfill
    \begin{center}
    \rule{1.0\textwidth}{1pt}\par
    \noindent\Huge\textbf{\@title}\par
    \ifx\@subtitle\@empty\else\bigskip\noindent\LARGE\@subtitle\par\fi
    \bigskip\bigskip
    \Large\tfgitic@location, \today\par
    \rule{1.0\textwidth}{1pt}\par
    \end{center}

    \vfill

    \begin{center}
      \LARGE
      {\scriptsize\tfgitic@submittext}\\
      \textsc{\tfgitic@author}\\
      {\scriptsize\tfgitic@fulfilltext}\\
      \tfgitic@degree
      \par
      \bigskip\bigskip\bigskip\Large
      \tfgitic@advisorblock\\
    \end{center}
    \vfill
    \begin{center}\tfgitic@topicsname: \tfgitic@topics\end{center}
    \newpage
    \thispagestyle{empty}
    \if@creativecommons
    \begin{minipage}{1.0\linewidth}
      \iflanguage{english}{
        \large This work is licensed under the Creative Commons
        Attribution-NonCommercial-ShareAlike 4.0 License. To view
        a copy of this license, visit
        \url{https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode}
        or send a letter to Creative Commons, PO Box 1866,
        Mountain View, CA 94042, USA.}{}
      \iflanguage{catalan}{
        \large Aquesta obra està subjecta a una llicència
        Attribution-NonCommercial-ShareAlike 4.0 de Creative
        Commons. Per veure'n una còpia, visiteu
        \url{https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.es}
        o envieu una carta a Creative Commons, 171 Second Street, Suite
        300, San Francisco, California 94105, USA.}{}
    \end{minipage}
    \else
    \mbox{}
    \fi
    \vfill
    {\tfgitic@typohow}
  \end{titlepage}
}

% Pàgines de la dedicatòria i els agraïments
\newcommand*{\tfgitic@dedicacio}{
  \ifx\@dedication\empty\else
  \begin{titlepage}
    \mbox{}
    \vskip 3cm
    \begin{raggedleft}
      \large\@dedication\par
    \end{raggedleft}
    \vfill
  \end{titlepage}
  \fi
  %
  \ifx\ackmsg\empty\else
  \begin{titlepage}
    \mbox{}
    \vskip 3cm
    \begin{center}
      \Large
      \iflanguage{english}{\textbf{Acknowledgments}}{}
      \iflanguage{catalan}{\textbf{Agraïments}}{}
    \end{center}
    \medskip
    \ackmsg
  \end{titlepage}
  \fi
}%

% Pàgina de la declaració responsable de no-plagi
\newcommand{\tfgitic@declaracio}{
  \if@catalanlang
  \begin{titlepage}
    \mbox{}
    \vskip 3cm
    \begin{center}
      \textbf{\Large Declaració Responsable}
    \end{center}
    \bigskip\bigskip\noindent
    Jo, \tfgitic@author{}, \tfgitic@authorname{} d'aquest Treball Final \if@master de Màster\else de Grau\fi,
    \par\bigskip\noindent
    DECLARO
    \par\bigskip\noindent
    Que aquest treball i la memòria que l'acompanya són originals
    i fruit exclusivament de la meva feina. Així mateix, totes les
    fonts consultades s'han recollit a la bibliografia.
    \par\medskip\noindent
    En sotmetre aquesta memòria a la seva avaluació, dono per
    signada aquesta declaració a efectes del que recull
    l'acord CG/2019/05/10, de 8 d'octubre de 2019, del Consell
    de Govern de la Universitat Politècnica de Catalunya, pel
    qual s'aprova el procediment de prevenció de plagi.
    \par\bigskip\noindent
    \tfgitic@location, \today{}
    \par
    \vfill
  \end{titlepage}
  \else
  \begin{titlepage}
    \hbox{}
    \vskip 3cm
    \begin{center}
      \textbf{\Large Declaration of Responsibility}
    \end{center}
    \bigskip\bigskip\noindent
    I, \tfgitic@author{}, \tfgitic@authorname{} of this \if@master Master's Thesis\else Capstone Project\fi,
    \par\bigskip\noindent
    DECLARE
    \par\bigskip\noindent
    That this project and the accompanying report are original
    and solely the result of my work. Furthermore, all sources
    consulted have been included in the bibliography.
    \par\medskip\noindent
    By submitting this report for evaluation, I consider this
    declaration signed for the purposes outlined in the agreement
    CG/2019/05/10, dated October 8, 2019, of the Governing Council
    of the Universitat Politècnica de Catalunya, which approves
    the procedure for plagiarism prevention.
    \par\bigskip\noindent
    \tfgitic@location, \today{}
    \par\vfill
  \end{titlepage}
  \fi
  \newpage
}


% estil de pagina
\pagestyle{scrheadings}
\setkomafont{pageheadfoot}{\normalfont\rmfamily\itshape\scriptsize}
\setkomafont{pagenumber}{\normalfont\rmfamily\itshape\normalsize}


% enumeracions amb parèntesi (només en català)
\iflanguage{catalan}{
  \renewcommand{\labelenumi}{\arabic{enumi})}
  \renewcommand{\labelenumii}{\alph{enumii})}
  \renewcommand{\labelenumiii}{\roman{enumiii})}
}

\newcommand{\acro}[1]{\textsc{#1}}
\newcommand{\est}[1]{\textit{#1}}
\let\fitx\nolinkurl
\newverbcommand{\ord}{}{}

% Agraïments
\def\ackmsg{}
\NewEnviron{acknowledgments}{\xdef\ackmsg{\unexpanded\expandafter{\BODY}}}
% Resum en anglès
\def\abstracten{}
\NewEnviron{abstract}{\xdef\abstracten{\unexpanded\expandafter{\BODY}}}
% Resum en català
\def\abstractca{}
\NewEnviron{resum}{\xdef\abstractca{\unexpanded\expandafter{\BODY}}}


% Preferències pels floats
\def\fps@figure{tbp}
\def\fps@table{tbp}

% Fonts especials
\setkomafont{descriptionlabel}{\sffamily}
\setkomafont{caption}{\small}
\setkomafont{captionlabel}{\usekomafont{caption}}


%%
%% Callback d'inici de document:
%% crea automàticament totes les pàgines normatives inicials
%%

\AtBeginDocument{
  % nasty trick to avoid:
  % https://www.mail-archive.com/tex-hyphen@tug.org/msg01801.html
  \if@catalanlang\lccode`'=0\fi
  \sloppy
  \maketitle{}
  \frontmatter
  \tfgitic@declaracio
  \tfgitic@dedicacio
  \if@catalanlang
    \chapter{Resum}
    \abstractca
    {\let\clearpage\relax\chapter{Abstract}}
    \begin{otherlanguage}{english}\noindent\abstracten\end{otherlanguage}
  \else
    \chapter{Abstract}
    \abstracten
    {\let\clearpage\relax\chapter{Resum}}
    \begin{otherlanguage}{catalan}\noindent\abstractca\end{otherlanguage}
  \fi
  \tableofcontents
  \mainmatter
}
