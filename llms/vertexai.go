package llms

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"

	aiplatform "cloud.google.com/go/aiplatform/apiv1"
	"cloud.google.com/go/aiplatform/apiv1/aiplatformpb"
	"cloud.google.com/go/vertexai/genai"
	"github.com/binarycraft007/fast-graphrag-go/llms/vertexai"
	"github.com/binarycraft007/fast-graphrag-go/types"
	"google.golang.org/api/option"
	"google.golang.org/protobuf/types/known/structpb"
)

// OpenAILLMService implements the LLMService interface for OpenAI.
type VertexAILLMService struct {
	Config          *MessageConfig
	APIKey          string
	MaxRetries      int
	Client          *genai.Client
	EmbeddingClient *aiplatform.PredictionClient
	MaxTokens       int
	EmbeddingDim    int
}

func DefaultVertexAILLMOptions() *MessageConfig {
	return &MessageConfig{
		Model:        "gemini-1.5-flash-002",
		MaxTokens:    8000,
		ResponseType: reflect.TypeOf(""),
		EmbeddingDim: 768,
	}
}

// NewOpenAILLMService initializes an OpenAI-based LLM service.
func NewVertexAILLMService(ctx context.Context, options ...MessageOptions) (*VertexAILLMService, error) {
	config := DefaultVertexAILLMOptions()
	for _, opt := range options {
		opt(config)
	}
	client, err := genai.NewClient(ctx, config.ProjectID, config.Location)
	if err != nil {
		return nil, err
	}

	apiEndpoint := fmt.Sprintf("%s-aiplatform.googleapis.com:443", config.Location)
	embeddingClient, err := aiplatform.NewPredictionClient(ctx, option.WithEndpoint(apiEndpoint))
	if err != nil {
		return nil, err
	}
	return &VertexAILLMService{
		Config:          config,
		Client:          client,
		EmbeddingClient: embeddingClient,
		MaxRetries:      3,
	}, nil
}

// SendMessage sends a message to the language model and receives a response.
func (g *VertexAILLMService) SendMessage(ctx context.Context, prompt string, options ...MessageOptions) (any, error) {
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

		g.Config.HistoryMessages = append(g.Config.HistoryMessages, resp.Candidates[0])

		var output string
		for _, part := range resp.Candidates[0].Content.Parts {
			if text, ok := part.(genai.Text); ok {
				output += string(text)
			}
		}
		return output, nil
	case reflect.Slice, reflect.Array, reflect.Struct:
		schema, err := vertexai.GenerateSchemaFromType(g.Config.ResponseType)
		if err != nil {
			return nil, err
		}
		genaiModel.GenerationConfig.ResponseMIMEType = "application/json"
		genaiModel.GenerationConfig.ResponseSchema = schema
		resp, err := cs.SendMessage(ctx, genai.Text(prompt))
		if err != nil {
			return nil, err
		}

		g.Config.HistoryMessages = append(g.Config.HistoryMessages, resp.Candidates[0])

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
func (g *VertexAILLMService) GetEmbedding(ctx context.Context, texts []string, options ...MessageOptions) ([]Embedding, error) {
	for _, opt := range options {
		opt(g.Config)
	}

	chunks := chunkTexts(texts, g.MaxTokens*types.TOKEN_TO_CHAR_RATIO)

	var embeddings []Embedding
	for _, chunk := range chunks {
		embedds, err := g.embedTexts(ctx, chunk)
		if err != nil {
			return nil, err
		}
		embeddings = append(embeddings, embedds...)
	}

	return embeddings, nil
}

// embedTexts shows how embeddings are set for text-embedding-005 model
func (g *VertexAILLMService) embedTexts(ctx context.Context, texts []string) ([]Embedding, error) {
	endpoint := fmt.Sprintf(
		"projects/%s/locations/%s/publishers/google/models/%s",
		g.Config.ProjectID,
		g.Config.Location,
		g.Config.Model,
	)
	instances := make([]*structpb.Value, len(texts))
	for i, text := range texts {
		instances[i] = structpb.NewStructValue(&structpb.Struct{
			Fields: map[string]*structpb.Value{
				"content":   structpb.NewStringValue(text),
				"task_type": structpb.NewStringValue("QUESTION_ANSWERING"),
			},
		})
	}

	params := structpb.NewStructValue(&structpb.Struct{
		Fields: map[string]*structpb.Value{
			"outputDimensionality": structpb.NewNumberValue(float64(g.Config.EmbeddingDim)),
		},
	})

	req := &aiplatformpb.PredictRequest{
		Endpoint:   endpoint,
		Instances:  instances,
		Parameters: params,
	}
	resp, err := g.EmbeddingClient.Predict(ctx, req)
	if err != nil {
		return nil, err
	}

	embeddings := make([]Embedding, len(resp.Predictions))
	for i, prediction := range resp.Predictions {
		values := prediction.GetStructValue().Fields["embeddings"].GetStructValue().Fields["values"].GetListValue().Values
		embeddings[i].Vector = make([]float32, len(values))
		for j, value := range values {
			embeddings[i].Vector[j] = float32(value.GetNumberValue())
		}
	}
	return embeddings, nil
}
