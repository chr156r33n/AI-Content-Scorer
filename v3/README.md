# Text Analysis App

A Streamlit application that analyzes text passages for semantic triplets, hedging language, topic drift, and overlong passages. The app provides interactive highlighting and detailed metrics to help improve text quality.

## Features

### Core Analysis
- **Semantic Triplets (Subject-Predicate-Object)**: Extracts and highlights S-P-O relationships using spaCy dependency parsing
- **Hedging Language Detection**: Identifies uncertain or cautious language with customizable lexicon
- **Topic Drift Detection**: Uses TF-IDF and cosine similarity to detect topic changes between sentences
- **Overlong Passage Detection**: Identifies sentences and paragraphs that exceed configurable word thresholds

### Interactive Features
- **Real-time Highlighting**: Color-coded inline highlighting of detected features
- **Configurable Thresholds**: Adjustable parameters for all detection algorithms
- **Editable Hedging Lexicon**: Customize the list of hedging terms
- **Readability Metrics**: Optional textstat integration for readability analysis
- **Detailed Results**: Expandable sections with comprehensive analysis breakdown

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or download the project files**
   ```bash
   # If you have the files in a directory
   cd /path/to/your/project
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy English model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Run the application**
   ```bash
   streamlit run text_analysis_app.py
   ```

5. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, navigate to the URL shown in your terminal

## Usage

### Basic Workflow

1. **Input Text**: Paste your text into the main text area
2. **Configure Settings**: Adjust thresholds and hedging terms in the sidebar
3. **Analyze**: Click "🔍 Analyze Text" to process your text
4. **Review Results**: Examine highlighted text and detailed metrics
5. **Iterate**: Adjust settings and re-analyze as needed

### Configuration Options

#### Hedging Terms
- Edit the list of hedging terms in the sidebar
- Default terms include: "might", "may", "could", "suggests", "possibly", etc.
- Add or remove terms based on your specific needs

#### Thresholds
- **Sentence Word Threshold**: Maximum words per sentence (default: 35)
- **Paragraph Word Threshold**: Maximum words per paragraph (default: 180)
- **Topic Drift Threshold**: Cosine similarity threshold for drift detection (default: 0.3)

### Understanding the Output

#### Color Coding
- 🟢 **Green**: Subject (in semantic triplets)
- 🔵 **Cyan**: Predicate/Verb (in semantic triplets)
- 🟣 **Purple**: Object (in semantic triplets)
- 🟡 **Amber**: Hedging language
- 🔵 **Blue**: Topic drift sentences
- 🔴 **Red**: Overlong passages

#### Metrics Panel
- **Sentences**: Total sentence count
- **Paragraphs**: Total paragraph count
- **Hedging instances**: Number of hedging terms found
- **S-P-O triplets**: Number of semantic triplets extracted
- **Topic drift**: Number of sentences with topic drift
- **Overlong spans**: Number of overlong sentences/paragraphs

#### Readability Metrics (if textstat is available)
- **Flesch Reading Ease**: Readability score (0-100, higher = easier)
- **Flesch-Kincaid Grade**: U.S. grade level
- **Gunning Fog**: Years of education needed
- **SMOG Index**: Reading grade level
- **Automated Readability Index**: Character-based readability

## Technical Details

### Dependencies
- `streamlit`: Web application framework
- `spacy`: Natural language processing
- `scikit-learn`: TF-IDF and cosine similarity
- `textstat`: Readability metrics (optional)
- `numpy`: Numerical computations
- `pandas`: Data manipulation

### Architecture
- **Frontend**: Streamlit web interface
- **NLP Engine**: spaCy for dependency parsing and sentence segmentation
- **Similarity**: scikit-learn TF-IDF vectorization
- **Highlighting**: Custom HTML generation with inline styles
- **Caching**: Streamlit caching for spaCy model loading

### Performance Considerations
- Uses lightweight spaCy model (`en_core_web_sm`)
- No external API calls or heavy transformer models
- Optimized for interactive use with reasonable text lengths
- Caching reduces model loading time

## Customization

### Adding New Detection Features
1. Create a new detection function following the existing pattern
2. Add configuration options to the sidebar
3. Update the highlighting system with new color codes
4. Integrate results into the summary panel

### Modifying Highlighting
- Edit the `color_map` dictionary in `highlight_text()` function
- Adjust CSS styles for different highlight types
- Add new highlight categories as needed

### Extending Hedging Detection
- Modify the `detect_hedging()` function for more sophisticated patterns
- Add regex patterns for complex hedging constructions
- Implement context-aware hedging detection

## Troubleshooting

### Common Issues

1. **spaCy model not found**
   ```
   Error: spaCy English model not found
   ```
   **Solution**: Run `python -m spacy download en_core_web_sm`

2. **Import errors**
   ```
   ModuleNotFoundError: No module named 'spacy'
   ```
   **Solution**: Install dependencies with `pip install -r requirements.txt`

3. **Performance issues with long texts**
   - Consider breaking very long texts into smaller chunks
   - Adjust window sizes for topic drift detection
   - Use sentence-level analysis for better performance

4. **Highlighting not working**
   - Check that text contains the expected patterns
   - Verify threshold settings are appropriate
   - Ensure spaCy model is loaded correctly

### Getting Help
- Check the Streamlit documentation for UI issues
- Review spaCy documentation for NLP-related problems
- Examine the console output for error messages

## Future Enhancements

### Planned Features
- Export annotations as CSV/JSON
- Batch file processing
- Custom spaCy Matcher patterns for triplets
- SBERT embeddings option for drift detection
- Advanced readability analysis
- Multi-language support

### Contributing
- Fork the repository
- Create feature branches
- Follow existing code style
- Add tests for new functionality
- Submit pull requests with clear descriptions

## License

This project is open source and available under the MIT License.