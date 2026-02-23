// All TypeScript interfaces for SoulMatch

export interface Character {
  id: string;
  name: string;
  avatar: string;
  job: string;
  city: string;
  age: number;
  status: 'Online' | 'Away' | 'Busy';
  interests: string[];
  bio: string;
}

export interface BotInfo {
  profile_id: string;
  age: number | null;
  sex: string | null;
  location: string | null;
  communication_style: string;
  core_values: string[];
  interests: string[];
  relationship_goals: string;
  personality_summary: string;
}

export interface Message {
  id: string;
  sender: 'user' | 'bot' | 'system';
  content: string;
  timestamp: Date;
}

export interface EmotionState {
  emotion: string;
  confidence: number;
  intensity: number;
}

export interface WarningState {
  level: string;
  message: string;
  risk_score: number;
}

export interface FeatureData {
  features: Record<string, any>;
  confidences: Record<string, number>;
  turn: number;
  low_confidence: string[];
  average_confidence: number;
  conformal?: ConformalData;
}

export interface ConformalData {
  coverage_guarantee: number;
  avg_set_size: number;
  singletons: number;
  total_dims: number;
  prediction_sets: Record<string, PredictionSet>;
}

export interface PredictionSet {
  set: string[];
  point: string;
  size: number;
  llm_conf: number;
  calibrated_conf: number;
}

export interface MemoryItem {
  content: string;
  relevance?: number;
}

export interface EmotionEntry {
  turn: number;
  emotion: string;
  intensity: number;
}

export interface EpisodicMemory {
  episode_id: string;
  turn_range: number[];
  summary: string;
  key_events: string[];
  emotion_trend: string;
}

export interface SemanticMemory {
  reflection_id: string;
  turn_range: number[];
  reflection: string;
  feature_updates: Record<string, any>;
  relationship_insights: string;
}

export interface MemoryStats {
  current_turn: number;
  working_memory_size: number;
  episodic_memory_count: number;
  semantic_memory_count: number;
  compression_ratio: number;
  episodic_memories?: EpisodicMemory[];
  semantic_memories?: SemanticMemory[];
}

export interface SocialVote {
  agent: string;
  vote: string;
  rel_status: string;
  confidence: number;
  reasoning: string;
  key_factors?: string[];
  demographics?: { age?: number; gender?: string; relationship_status?: string };
}

export interface RelationshipData {
  rel_status: string;
  rel_type: string;
  sentiment: string;
  can_advance: boolean;
  advance_prediction_set: string[];
  social_votes?: SocialVote[];
  vote_distribution?: Record<string, number>;
}

export interface MilestoneReport {
  turn: number;
  type: string;
  message: string;
  predicted_status_at_turn_30?: string;
  final_status?: string;
  avg_trust_score?: number;
}

export interface TrustPoint {
  turn: number;
  trust: number;
}

export interface ContextData {
  state?: string;
  risk_level?: string;
  turn_count?: number;
  user_emotion?: string;
  avg_feature_confidence?: number;
}
