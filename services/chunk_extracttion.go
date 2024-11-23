package services

import (
	"regexp"
	"strings"

	"github.com/binarycraft007/fast-graphrag-go/types"
)

var DEFAULT_SEPARATORS = []string{
	// Paragraph and page separators
	"\n\n\n",
	"\n\n",
	"\r\n\r\n",
	// Sentence ending punctuation
	"。", // Chinese period
	"．", // Full-width dot
	".", // English period
	"！", // Chinese exclamation mark
	"!", // English exclamation mark
	"？", // Chinese question mark
	"?", // English question mark
}

// Config for the chunking service
type DefaultChunkingServiceConfig struct {
	Separators        []string
	ChunkTokenSize    int
	ChunkTokenOverlap int
}

// Constructor for the config with defaults
func NewDefaultChunkingServiceConfig() DefaultChunkingServiceConfig {
	return DefaultChunkingServiceConfig{
		Separators:        DEFAULT_SEPARATORS,
		ChunkTokenSize:    800,
		ChunkTokenOverlap: 100,
	}
}

// BaseChunkingService interface defines the behavior of chunking services
type BaseChunkingService interface {
	Extract(data []types.Document) [][]types.Chunk
}

// DefaultChunkingService implements chunking logic
type DefaultChunkingService struct {
	Config       DefaultChunkingServiceConfig
	splitRe      *regexp.Regexp
	chunkSize    int
	chunkOverlap int
}

// Constructor for DefaultChunkingService
func NewDefaultChunkingService() *DefaultChunkingService {
	config := NewDefaultChunkingServiceConfig()
	pattern := "(" + strings.Join(escapeSeparators(config.Separators), "|") + ")"
	return &DefaultChunkingService{
		Config:       config,
		splitRe:      regexp.MustCompile(pattern),
		chunkSize:    config.ChunkTokenSize * types.TOKEN_TO_CHAR_RATIO,
		chunkOverlap: config.ChunkTokenOverlap * types.TOKEN_TO_CHAR_RATIO,
	}
}

// escapeSeparators escapes separators for regex
func escapeSeparators(separators []string) []string {
	escaped := make([]string, len(separators))
	for i, sep := range separators {
		escaped[i] = regexp.QuoteMeta(sep)
	}
	return escaped
}

// Extract unique chunks from data
func (s *DefaultChunkingService) Extract(data []types.Document) [][]types.Chunk {
	chunksPerData := make([][]types.Chunk, 0, len(data))
	for _, d := range data {
		uniqueChunkIDs := make(map[uint64]struct{})
		extractedChunks := s.extractChunks(d)
		chunks := make([]types.Chunk, 0, len(extractedChunks))
		for _, chunk := range extractedChunks {
			if _, exists := uniqueChunkIDs[chunk.ID]; !exists {
				uniqueChunkIDs[chunk.ID] = struct{}{}
				chunks = append(chunks, chunk)
			}
		}
		chunksPerData = append(chunksPerData, chunks)
	}
	return chunksPerData
}

// Extract chunks from a single document
func (s *DefaultChunkingService) extractChunks(data types.Document) []types.Chunk {
	data.Data = sanitizeInput(data.Data)
	var chunks []string
	if len(data.Data) <= s.chunkSize {
		chunks = []string{data.Data}
	} else {
		chunks = s.splitText(data.Data)
	}

	result := make([]types.Chunk, len(chunks))
	for i, chunk := range chunks {
		h := xxhash64{}
		h.update([]byte(chunk))
		result[i] = types.Chunk{
			ID:       h.digest(),
			Content:  chunk,
			Metadata: data.Metadata,
		}
	}
	return result
}

// Sanitize input data by removing unwanted characters
func sanitizeInput(input string) string {
	return regexp.MustCompile(`[\x00-\x1f\x7f-\x9f]`).ReplaceAllString(input, "")
}

// Split text into chunks based on separators
func (s *DefaultChunkingService) splitText(text string) []string {
	splits := s.splitRe.Split(text, -1)
	return s.mergeSplits(splits)
}

// Merge splits into chunks
func (s *DefaultChunkingService) mergeSplits(splits []string) []string {
	if len(splits) == 0 {
		return []string{}
	}

	splits = append(splits, "") // Ensure a trailing separator
	mergedSplits := [][]string{}
	currentChunk := []string{}
	currentChunkLength := 0

	for i, split := range splits {
		splitLength := len(split)
		if i%2 == 1 || currentChunkLength+splitLength <= s.chunkSize-s.chunkOverlap {
			currentChunk = append(currentChunk, split)
			currentChunkLength += splitLength
		} else {
			mergedSplits = append(mergedSplits, currentChunk)
			currentChunk = []string{split}
			currentChunkLength = splitLength
		}
	}
	mergedSplits = append(mergedSplits, currentChunk)

	if s.chunkOverlap > 0 {
		return s.enforceOverlap(mergedSplits)
	}

	return flattenChunks(mergedSplits)
}

// Enforce overlap between chunks
func (s *DefaultChunkingService) enforceOverlap(chunks [][]string) []string {
	result := []string{}
	for i, chunk := range chunks {
		if i == 0 {
			result = append(result, strings.Join(chunk, ""))
		} else {
			overlap := getOverlap(chunks[i-1], s.chunkOverlap)
			result = append(result, overlap+strings.Join(chunk, ""))
		}
	}
	return result
}

// Get overlap from previous chunk
func getOverlap(prevChunk []string, overlapSize int) string {
	overlap := ""
	length := 0
	for i := len(prevChunk) - 1; i >= 0 && length < overlapSize; i-- {
		length += len(prevChunk[i])
		overlap = prevChunk[i] + overlap
	}
	return overlap
}

// Flatten chunk lists into strings
func flattenChunks(chunks [][]string) []string {
	result := make([]string, len(chunks))
	for i, chunk := range chunks {
		result[i] = strings.Join(chunk, "")
	}
	return result
}
