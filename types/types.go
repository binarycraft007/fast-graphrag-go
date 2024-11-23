package types

// Constants
const TOKEN_TO_CHAR_RATIO = 4

// Chunk represents a chunk of data
type Chunk struct {
	ID       uint64
	Content  string
	Metadata map[string]interface{}
}

// Document represents an input document
type Document struct {
	Data     string
	Metadata map[string]interface{}
}

// Entity represents an entity in the graph.
type Entity struct {
	Name        string `json:"name"`
	Type        string `json:"type"`
	Description string `json:"description"`
}

// TGraph represents a graph with entities and relationships.
type Graph struct {
	Entities           []Entity   `json:"entities"`
	Relationships      []Relation `json:"relationships"`
	OtherRelationships []Relation `json:"other_relationships"`
}

// TRelation represents a relationship in the graph.
type Relation struct {
	Source      string   `json:"source"`
	Target      string   `json:"target"`
	Description string   `json:"description"`
	Chunks      []uint64 `json:"chunks"`
}
