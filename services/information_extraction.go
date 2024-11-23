package services

import (
	"context"
	"errors"
	"log"
	"reflect"
	"strings"
	"sync"

	"github.com/binarycraft007/fast-graphrag-go/llms"
	"github.com/binarycraft007/fast-graphrag-go/prompts"
	"github.com/binarycraft007/fast-graphrag-go/types"
)

// BaseGraphStorage defines the interface for graph storage.
type BaseGraphStorage[Node, Edge, ID any] interface {
	InsertStart() error
	InsertDone() error
}

// BaseGraphUpsertPolicy defines the interface for graph upserting logic.
type BaseGraphUpsertPolicy[Node, Edge, ID any] interface {
	Upsert(llm BaseLLMService, storage BaseGraphStorage[Node, Edge, ID], nodes []Node, edges []Edge) error
}

// BaseLLMService defines the interface for LLM services.
type BaseLLMService interface {
	FormatAndSendPrompt(promptKey string, formatArgs map[string]string, responseModel interface{}) error
}

// GleaningStatus represents the status of gleaning.
type Status string

const (
	Done     Status = "done"
	Continue Status = "continue"
)

type GleaningStatus struct {
	Status Status `json:"status"`
}

// BaseInformationExtractionService defines the base for information extraction services.
type BaseInformationExtractionService[Chunk, Node, Edge, ID any] struct {
	GraphUpsert      BaseGraphUpsertPolicy[Node, Edge, ID]
	MaxGleaningSteps int
}

// Extract extracts entities and relationships from documents.
func (s *BaseInformationExtractionService[Chunk, Node, Edge, ID]) Extract(
	llm BaseLLMService,
	documents [][]Chunk,
	promptArgs map[string]string,
	entityTypes []string,
) ([]chan *BaseGraphStorage[Node, Edge, ID], error) {
	return nil, errors.New("not implemented")
}

// ExtractEntitiesFromQuery extracts entities from a query string.
func (s *BaseInformationExtractionService[Chunk, Node, Edge, ID]) ExtractEntitiesFromQuery(
	llm BaseLLMService, query string, promptArgs map[string]string,
) ([]types.Entity, error) {
	return nil, errors.New("not implemented")
}

// DefaultInformationExtractionService implements the default information extraction.
type DefaultInformationExtractionService struct {
	BaseInformationExtractionService[types.Chunk, types.Entity, types.Relation, string]
}

// Extract extracts both entities and relationships.
func (s *DefaultInformationExtractionService) Extract(
	llm llms.LLMService,
	documents [][]types.Chunk,
	promptArgs map[string]any,
	entityTypes []string,
) ([]chan *BaseGraphStorage[types.Entity, types.Relation, string], error) {
	results := make([]chan *BaseGraphStorage[types.Entity, types.Relation, string], len(documents))
	for i, document := range documents {
		results[i] = make(chan *BaseGraphStorage[types.Entity, types.Relation, string], 1)
		go func(doc []types.Chunk, result chan *BaseGraphStorage[types.Entity, types.Relation, string]) {
			graph, err := s.extractChunks(llm, doc, promptArgs, entityTypes)
			if err != nil {
				log.Println("Error extracting chunks:", err)
				close(result)
				return
			}
			result <- graph
			close(result)
		}(document, results[i])
	}
	return results, nil
}

func (s *DefaultInformationExtractionService) extractChunks(
	llm llms.LLMService, chunks []types.Chunk, promptArgs map[string]any, entityTypes []string,
) (*BaseGraphStorage[types.Entity, types.Relation, string], error) {
	var wg sync.WaitGroup
	var mu sync.Mutex
	chunkResults := make([]*types.Graph, len(chunks))
	errors := make([]error, 0)

	wg.Add(len(chunks))
	for i, chunk := range chunks {
		go func(idx int, c types.Chunk) {
			defer wg.Done()
			log.Println("extracting chunk:", c.ID)
			graph, err := s.extractChunk(llm, c, promptArgs, entityTypes)
			mu.Lock()
			if err != nil {
				errors = append(errors, err)
			} else {
				chunkResults[idx] = graph
			}
			mu.Unlock()
		}(i, chunk)
	}
	wg.Wait()

	if len(errors) > 0 {
		return nil, errors[0] // Return the first error encountered.
	}

	return s.mergeGraphs(llm, chunkResults)
}

func (s *DefaultInformationExtractionService) extractChunk(
	llm llms.LLMService, chunk types.Chunk, promptArgs map[string]any, entityTypes []string,
) (*types.Graph, error) {
	promptArgs["input_text"] = chunk.Content
	promptArgs["entity_relationship_extraction"] = prompts.EntityRelationshipExtractionExample

	ctx := context.Background()
	chunkGraph, err := llms.FormatAndSendPrompt(
		ctx,
		"entity_relationship_extraction",
		llm,
		promptArgs,
		llms.WithResponseType(reflect.TypeOf(types.Graph{})),
	)
	if err != nil {
		return nil, err
	}

	// Glean additional details if necessary
	finalGraph, err := s.gleaning(llm, chunkGraph.(*types.Graph), []map[string]string{})
	if err != nil {
		return nil, err
	}

	cleanEntityTypes := s.cleanEntityTypes(entityTypes)
	for i := range finalGraph.Entities {
		if !cleanEntityTypes[strings.ToUpper(strings.ReplaceAll(finalGraph.Entities[i].Type, " ", ""))] {
			finalGraph.Entities[i].Type = "UNKNOWN"
		}
	}

	for i := range finalGraph.Relationships {
		finalGraph.Relationships[i].Chunks = append(finalGraph.Relationships[i].Chunks, chunk.ID)
	}

	return finalGraph, nil
}

func (s *DefaultInformationExtractionService) gleaning(
	llm llms.LLMService, initialGraph *types.Graph, history []map[string]string,
) (*types.Graph, error) {
	currentGraph := initialGraph

	//for step := 0; step < s.MaxGleaningSteps; step++ {
	//	gleaningResult := &types.Graph{}
	//	err := llm.FormatAndSendPrompt("entity_relationship_continue_extraction", map[string]string{}, gleaningResult)
	//	if err != nil {
	//		log.Println("Gleaning error:", err)
	//		return nil, err
	//	}

	//	currentGraph.Entities = append(currentGraph.Entities, gleaningResult.Entities...)
	//	currentGraph.Relationships = append(currentGraph.Relationships, gleaningResult.Relationships...)

	//	gleaningStatus := &GleaningStatus{}
	//	err = llm.FormatAndSendPrompt("entity_relationship_gleaning_done_extraction", map[string]string{}, gleaningStatus)
	//	if err != nil {
	//		log.Println("Gleaning status error:", err)
	//		return nil, err
	//	}

	//	if gleaningStatus.Status == "done" {
	//		break
	//	}
	//}

	return currentGraph, nil
}

func (s *DefaultInformationExtractionService) mergeGraphs(
	llm llms.LLMService, graphs []*types.Graph,
) (*BaseGraphStorage[types.Entity, types.Relation, string], error) {
	// Simulate graph merging logic
	return nil, nil
}

func (s *DefaultInformationExtractionService) cleanEntityTypes(entityTypes []string) map[string]bool {
	cleaned := make(map[string]bool)
	for _, t := range entityTypes {
		cleaned[strings.ToUpper(strings.ReplaceAll(t, " ", ""))] = true
	}
	return cleaned
}
