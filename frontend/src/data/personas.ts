export interface Persona {
  id: number;
  name: string;
  emoji: string;
  isBot: boolean;
  profile: {
    age: number;
    gender: string;
    occupation: string;
    location: string;
    mbti?: string;
  };
  tags: string[];
}

// Based on speed dating dataset demographics and synthetic profiles
export const PERSONAS: Persona[] = [
  { id: 0, name: 'Alex', emoji: '👨‍💼', isBot: true, profile: { age: 28, gender: 'Male', occupation: 'Product Manager', location: 'San Francisco', mbti: 'ENTJ' }, tags: ['Tech', 'Startups', 'Fitness'] },
  { id: 1, name: 'Emma', emoji: '👩‍🎨', isBot: true, profile: { age: 25, gender: 'Female', occupation: 'Designer', location: 'New York', mbti: 'INFP' }, tags: ['Art', 'Travel', 'Coffee'] },
  { id: 2, name: 'Sarah', emoji: '👩‍💻', isBot: true, profile: { age: 26, gender: 'Female', occupation: 'Software Engineer', location: 'Seattle', mbti: 'INTJ' }, tags: ['Coding', 'Gaming', 'Anime'] },
  { id: 3, name: 'David', emoji: '👨‍🔬', isBot: true, profile: { age: 32, gender: 'Male', occupation: 'Data Scientist', location: 'Boston', mbti: 'INTP' }, tags: ['AI', 'Math', 'Music'] },
  { id: 4, name: 'Mike', emoji: '👨‍🍳', isBot: true, profile: { age: 35, gender: 'Male', occupation: 'Restaurant Owner', location: 'Chicago', mbti: 'ESFP' }, tags: ['Food', 'Business', 'Travel'] },
  { id: 5, name: 'Lily', emoji: '👩‍🎤', isBot: true, profile: { age: 23, gender: 'Female', occupation: 'Musician', location: 'Los Angeles', mbti: 'ENFP' }, tags: ['Music', 'Photography', 'Arts'] },
  { id: 6, name: 'Jessica', emoji: '👩‍🏫', isBot: true, profile: { age: 27, gender: 'Female', occupation: 'Teacher', location: 'Austin', mbti: 'ESFJ' }, tags: ['Education', 'Travel', 'Movies'] },
  { id: 7, name: 'Ryan', emoji: '👨‍🎨', isBot: true, profile: { age: 30, gender: 'Male', occupation: 'Illustrator', location: 'Portland', mbti: 'ISFP' }, tags: ['Drawing', 'Cats', 'Indie Music'] },
  { id: 8, name: 'Tom', emoji: '👨‍🔧', isBot: true, profile: { age: 38, gender: 'Male', occupation: 'Engineer', location: 'Denver', mbti: 'ISTJ' }, tags: ['Mechanics', 'Fishing', 'Photography'] },
  { id: 9, name: 'Chris', emoji: '👨‍✈️', isBot: true, profile: { age: 31, gender: 'Male', occupation: 'Pilot', location: 'Miami', mbti: 'ESTP' }, tags: ['Flying', 'Extreme Sports', 'Travel'] },
];
