// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyAg20DJMkoLsmm92ZlAA3xwzzWvb99jgUA",
  authDomain: "polaigraph.firebaseapp.com",
  projectId: "polaigraph",
  storageBucket: "polaigraph.firebasestorage.app",
  messagingSenderId: "29482655484",
  appId: "1:29482655484:web:7b8ee060c577ee76198179",
  measurementId: "G-Z7DNXHYTZH"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);

export {auth, provider, signInWithPopup, db, storage };