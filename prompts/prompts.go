package prompts

import (
	_ "embed"
)

//go:embed entity_relationship_extraction.md
var entity_relationship_extraction string

//go:embed entity_relationship_extraction.json
var EntityRelationshipExtractionExample string

var Prompts = map[string]string{
	"entity_relationship_extraction": entity_relationship_extraction,
}
