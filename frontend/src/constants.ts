import type { Character } from './types';

export const CHARACTERS: Character[] = [
  { id: 'bot_0', name: 'Mina', avatar: 'https://api.dicebear.com/9.x/notionists/svg?seed=Mina', job: 'Policy Analyst', city: 'San Francisco', age: 29, status: 'Away', interests: ['Food', 'Travel', 'Books'], bio: 'Exploring the world one policy at a time' },
  { id: 'bot_1', name: 'Jade', avatar: 'https://api.dicebear.com/9.x/notionists/svg?seed=Jade', job: 'Office Admin', city: 'San Francisco', age: 25, status: 'Online', interests: ['Music', 'Food', 'Arts'], bio: 'Finding beauty in the everyday rhythm' },
  { id: 'bot_2', name: 'Sierra', avatar: 'https://api.dicebear.com/9.x/notionists/svg?seed=Sierra', job: 'Math Teacher', city: 'Oakland', age: 34, status: 'Away', interests: ['Travel', 'Outdoors', 'Music'], bio: 'Numbers by day, mountains by weekend' },
  { id: 'bot_3', name: 'Kevin', avatar: 'https://api.dicebear.com/9.x/notionists/svg?seed=Kevin', job: 'Software Dev', city: 'San Francisco', age: 33, status: 'Online', interests: ['Music', 'Tech', 'Outdoors'], bio: 'Building things that matter' },
  { id: 'bot_4', name: 'Marcus', avatar: 'https://api.dicebear.com/9.x/notionists/svg?seed=Marcus', job: 'VP Operations', city: 'Castro Valley', age: 33, status: 'Online', interests: ['Music', 'Food', 'Sports'], bio: 'Leading teams, chasing sunsets' },
  { id: 'bot_5', name: 'Derek', avatar: 'https://api.dicebear.com/9.x/notionists/svg?seed=Derek', job: 'Financial Analyst', city: 'Oakland', age: 39, status: 'Busy', interests: ['Food', 'Outdoors', 'Sports'], bio: 'Balancing spreadsheets and trail runs' },
  { id: 'bot_6', name: 'Luna', avatar: 'https://api.dicebear.com/9.x/notionists/svg?seed=Luna', job: 'Freelancer', city: 'Hayward', age: 26, status: 'Online', interests: ['Food', 'Travel', 'Music'], bio: 'Creating on my own terms' },
  { id: 'bot_7', name: 'Travis', avatar: 'https://api.dicebear.com/9.x/notionists/svg?seed=Travis', job: 'Sales Director', city: 'San Francisco', age: 42, status: 'Away', interests: ['Travel', 'Food', 'Outdoors'], bio: 'Closing deals, opening horizons' },
];

export const AGE_FILTERS = ['All', '20s', '30s', '40+'] as const;

export const EMOTION_EMOJI: Record<string, string> = {
  joy: '😄', sadness: '😢', anger: '😠', fear: '😨',
  surprise: '😲', disgust: '🤢', neutral: '😐', love: '😍',
  excitement: '🤩', anxiety: '😰',
};

export const STATUS_ORDER = ['stranger', 'acquaintance', 'crush', 'dating', 'committed'];
export const STATUS_LABELS: Record<string, string> = {
  stranger: 'STR', acquaintance: 'ACQ', crush: 'CRS', dating: 'DAT', committed: 'CMT',
};

export const STATUS_COLORS: Record<string, string> = {
  Online: '#39ff14',
  Away: '#ffb800',
  Busy: '#ff2e4c',
};
