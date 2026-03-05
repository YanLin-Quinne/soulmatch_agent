export interface Persona {
  id: number;
  name: string;
  age: number;
  gender: string;
  avatar: string;
  bio: string;
  personality: string[];
  interests: string[];
  occupation: string;
  isAI: boolean;
}

export const PERSONAS: Persona[] = [
  {
    id: 0,
    name: "Alex Chen",
    age: 28,
    gender: "Male",
    avatar: "👨‍💻",
    bio: "Software engineer passionate about AI and open source",
    personality: ["Analytical", "Curious", "Introverted", "Creative"],
    interests: ["Coding", "Gaming", "Sci-fi", "Coffee"],
    occupation: "Software Engineer",
    isAI: true
  },
  {
    id: 1,
    name: "Emma Wilson",
    age: 26,
    gender: "Female",
    avatar: "👩‍🎨",
    bio: "Digital artist exploring the intersection of art and technology",
    personality: ["Creative", "Empathetic", "Optimistic", "Spontaneous"],
    interests: ["Art", "Photography", "Travel", "Yoga"],
    occupation: "Digital Artist",
    isAI: true
  },
  {
    id: 2,
    name: "Marcus Johnson",
    age: 32,
    gender: "Male",
    avatar: "👨‍🏫",
    bio: "High school teacher who loves inspiring the next generation",
    personality: ["Patient", "Enthusiastic", "Organized", "Supportive"],
    interests: ["Education", "Reading", "Hiking", "Music"],
    occupation: "Teacher",
    isAI: true
  },
  {
    id: 3,
    name: "Sophia Lee",
    age: 24,
    gender: "Female",
    avatar: "👩‍🔬",
    bio: "Biotech researcher working on breakthrough therapies",
    personality: ["Methodical", "Ambitious", "Detail-oriented", "Rational"],
    interests: ["Science", "Research", "Running", "Podcasts"],
    occupation: "Researcher",
    isAI: true
  },
  {
    id: 4,
    name: "David Martinez",
    age: 30,
    gender: "Male",
    avatar: "👨‍🍳",
    bio: "Chef and food blogger sharing culinary adventures",
    personality: ["Passionate", "Adventurous", "Social", "Perfectionist"],
    interests: ["Cooking", "Food", "Wine", "Travel"],
    occupation: "Chef",
    isAI: true
  },
  {
    id: 5,
    name: "Olivia Brown",
    age: 27,
    gender: "Female",
    avatar: "👩‍⚕️",
    bio: "Pediatric nurse dedicated to caring for children",
    personality: ["Compassionate", "Reliable", "Warm", "Practical"],
    interests: ["Healthcare", "Volunteering", "Baking", "Gardening"],
    occupation: "Nurse",
    isAI: true
  },
  {
    id: 6,
    name: "Ryan Taylor",
    age: 29,
    gender: "Male",
    avatar: "👨‍💼",
    bio: "Marketing strategist helping brands tell their stories",
    personality: ["Charismatic", "Strategic", "Confident", "Adaptable"],
    interests: ["Marketing", "Networking", "Sports", "Startups"],
    occupation: "Marketing Manager",
    isAI: true
  },
  {
    id: 7,
    name: "Isabella Garcia",
    age: 25,
    gender: "Female",
    avatar: "👩‍🎓",
    bio: "Psychology PhD student researching human behavior",
    personality: ["Thoughtful", "Observant", "Intellectual", "Empathetic"],
    interests: ["Psychology", "Writing", "Meditation", "Theater"],
    occupation: "PhD Student",
    isAI: true
  },
  {
    id: 8,
    name: "James Anderson",
    age: 31,
    gender: "Male",
    avatar: "👨‍🔧",
    bio: "Mechanical engineer building sustainable solutions",
    personality: ["Practical", "Innovative", "Focused", "Humble"],
    interests: ["Engineering", "DIY", "Cycling", "Sustainability"],
    occupation: "Engineer",
    isAI: true
  },
  {
    id: 9,
    name: "Mia Thompson",
    age: 23,
    gender: "Female",
    avatar: "👩‍💻",
    bio: "UX designer creating delightful user experiences",
    personality: ["Creative", "User-focused", "Collaborative", "Curious"],
    interests: ["Design", "Tech", "Animation", "Coffee shops"],
    occupation: "UX Designer",
    isAI: true
  },
  {
    id: 10,
    name: "Sarah Kim",
    age: 28,
    gender: "Female",
    avatar: "👩‍🎤",
    bio: "Indie musician and songwriter",
    personality: ["Artistic", "Sensitive", "Passionate", "Free-spirited"],
    interests: ["Music", "Concerts", "Poetry", "Vintage fashion"],
    occupation: "Musician",
    isAI: false
  },
  {
    id: 11,
    name: "Tom Zhang",
    age: 33,
    gender: "Male",
    avatar: "👨‍✈️",
    bio: "Commercial pilot exploring the world",
    personality: ["Adventurous", "Calm", "Responsible", "Outgoing"],
    interests: ["Aviation", "Travel", "Photography", "Fitness"],
    occupation: "Pilot",
    isAI: false
  },
  {
    id: 12,
    name: "Nina Patel",
    age: 26,
    gender: "Female",
    avatar: "👩‍🏫",
    bio: "Environmental activist and blogger",
    personality: ["Passionate", "Idealistic", "Determined", "Articulate"],
    interests: ["Environment", "Activism", "Blogging", "Veganism"],
    occupation: "Activist",
    isAI: false
  },
  {
    id: 13,
    name: "Chris O'Brien",
    age: 29,
    gender: "Male",
    avatar: "👨‍🎨",
    bio: "Freelance photographer capturing life's moments",
    personality: ["Observant", "Patient", "Artistic", "Independent"],
    interests: ["Photography", "Nature", "Documentary", "Coffee"],
    occupation: "Photographer",
    isAI: false
  },
  {
    id: 14,
    name: "Lily Wang",
    age: 24,
    gender: "Female",
    avatar: "👩‍🚀",
    bio: "Aerospace engineer dreaming of Mars",
    personality: ["Ambitious", "Logical", "Driven", "Visionary"],
    interests: ["Space", "Science", "Robotics", "Sci-fi movies"],
    occupation: "Aerospace Engineer",
    isAI: false
  }
];
