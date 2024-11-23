package llms

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"

	"github.com/binarycraft007/fast-graphrag-go/llms/googleai"
	"github.com/binarycraft007/fast-graphrag-go/types"
	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

// OpenAILLMService implements the LLMService interface for OpenAI.
type GoogleAILLMService struct {
	Config       *MessageConfig
	APIKey       string
	MaxRetries   int
	Client       *genai.Client
	MaxTokens    int
	EmbeddingDim int
}

func DefaultGoogleAILLMOptions() *MessageConfig {
	return &MessageConfig{
		Model:        "gemini-1.5-flash-002",
		MaxTokens:    8000,
		ResponseType: reflect.TypeOf(""),
		EmbeddingDim: 768,
	}
}

// NewOpenAILLMService initializes an OpenAI-based LLM service.
func NewGoogleAILLMService(ctx context.Context, apiKey string, options ...MessageOptions) (*GoogleAILLMService, error) {
	config := DefaultGoogleAILLMOptions()
	for _, opt := range options {
		opt(config)
	}
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return nil, err
	}
	return &GoogleAILLMService{
		Config:     config,
		Client:     client,
		APIKey:     apiKey,
		MaxRetries: 3,
	}, nil
}

// SendMessage sends a message to the language model and receives a response.
func (g *GoogleAILLMService) SendMessage(ctx context.Context, prompt string, options ...MessageOptions) (any, error) {
	for _, opt := range options {
		opt(g.Config)
	}

	genaiModel := g.Client.GenerativeModel(g.Config.Model)
	genaiModel.SetCandidateCount(1)
	genaiModel.SetTemperature(0)
	genaiModel.SetMaxOutputTokens(int32(g.Config.MaxTokens))

	cs := genaiModel.StartChat()
	if g.Config.SystemPrompt != "" {
		genaiModel.SystemInstruction = &genai.Content{
			Role:  "system",
			Parts: []genai.Part{genai.Text(g.Config.SystemPrompt)},
		}
	}
	for _, content := range g.Config.HistoryMessages {
		cs.History = append(cs.History, content.(*genai.Content))
	}
	switch g.Config.ResponseType.Kind() {
	case reflect.String:
		resp, err := cs.SendMessage(ctx, genai.Text(prompt))
		if err != nil {
			return nil, err
		}
		var output string
		for _, part := range resp.Candidates[0].Content.Parts {
			if text, ok := part.(genai.Text); ok {
				output += string(text)
			}
		}
		return output, nil
	case reflect.Slice, reflect.Array, reflect.Struct:
		schema, err := googleai.GenerateSchemaFromType(g.Config.ResponseType)
		if err != nil {
			return nil, err
		}
		genaiModel.GenerationConfig.ResponseMIMEType = "application/json"
		genaiModel.GenerationConfig.ResponseSchema = schema
		resp, err := cs.SendMessage(ctx, genai.Text(prompt))
		if err != nil {
			return nil, err
		}

		var jsonData string
		for _, part := range resp.Candidates[0].Content.Parts {
			if text, ok := part.(genai.Text); ok {
				jsonData += string(text)
			}
		}

		// Create a new instance of the type using reflect.New
		instance := reflect.New(g.Config.ResponseType).Interface()

		// Unmarshal JSON into the dynamically created instance
		err = json.Unmarshal([]byte(jsonData), instance)
		if err != nil {
			return nil, fmt.Errorf("Error unmarshaling: %v", err)
		}
		return instance, nil
	default:
		return nil, errors.New("unsupported type")
	}
}

// GetEmbedding retrieves embeddings for the given texts.
func (g *GoogleAILLMService) GetEmbedding(ctx context.Context, texts []string, options ...MessageOptions) ([]Embedding, error) {
	for _, opt := range options {
		opt(g.Config)
	}

	chunks := chunkTexts(texts, g.MaxTokens*types.TOKEN_TO_CHAR_RATIO)

	model := g.Client.EmbeddingModel(g.Config.Model)
	batch := model.NewBatch()

	for _, chunk := range chunks {
		for _, text := range chunk {
			batch.AddContent(genai.Text(text))
		}
	}

	resp, err := model.BatchEmbedContents(ctx, batch)
	if err != nil {
		return nil, err
	}

	embeddings := make([]Embedding, len(resp.Embeddings))
	for i, embedding := range resp.Embeddings {
		copy(embeddings[i].Vector, embedding.Values)
	}

	return embeddings, nil
}
